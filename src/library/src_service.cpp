// #pragma once
#include <fractos/service/compute/cuda.hpp>


#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <./src_service.hpp>
#include <./src_device.hpp>
#include <./common.hpp>


using namespace fractos;
using namespace impl;
// using namespace fractos::service::compute::cuda;

core::future<core::cap::request>
impl::service::register_methods(std::shared_ptr<core::channel> ch)
{
    auto self = _self.lock();

    // Request handler for connect_serivce message
    auto reqb_connect_service =
        ch->make_request_builder<fractos::service::compute::cuda::message::connect_service::request>(
            ch->get_default_endpoint(),
            // Declare lambda to serve as the request receive handler. The
            // handler is called whenever the request (built with make_request()
            // below) is invoked.
            //
            // The handler receives the channel through which the invocation was
            // received (ch), and a unique_ptr to a receive buffer with the
            // immediate and capability arguments sent as part of the invocation
            // (args). The type of the arguments is taken from that passed to
            // channel::make_request_builder() above; i.e., args->imms has type
            // fractos::service::compute::cuda::message::connect_service::request::imms,
            // and args->caps has type
            // fractos::service::compute::cuda::message::connect_service::request::caps.
            //
            // The channel has as many receive buffers as set in
            // core::channel_config::recv_count. Deleting the buffer object will
            // delete any capability objects it received (unless you moved them
            // out of the received arguments), and will make the buffer
            // available to receive a new invocation. Note that failing to free
            // receive buffers will result in not receiving more invocations on
            // that channel, which can be used as a form of back pressure.
            //
            // Received capability arguments are associated to the channel where
            // the request invocation is received; i.e.,
            // args->caps_raw[x]->get_channel() == ch
            [self](auto ch, auto args) {
                self->handle_connect_service(ch, std::move(args));
            });

    // Request handler for make_device message
    auto reqb_make_device =
        ch->make_request_builder<fractos::service::compute::cuda::message::service::make_device::request>(
            ch->get_default_endpoint(),
            [self](auto ch, auto args) {
                self->handle_make_device(ch, std::move(args));
            });

    // Create the requests themselves, only using continuations to make it fully
    // asynchronous

    auto req_connect = std::make_shared<std::unique_ptr<core::cap::request>>();

    return reqb_connect_service
        .on_channel()
        .make_request()
        .then([req_connect, reqb_make_device, self](auto& fut) {
                  req_connect->reset(new core::cap::request(fut.get()));
                  return reqb_make_device
                      .on_channel()
                      .make_request()
                      .then([self](auto& fut) {
                                self->_req_make_device = fut.get();
                            });
              })
        .unwrap()
        // Return the connect_service request after all else is ready
        .then([req_connect](auto& fut) {
                  fut.get();
                  return std::move(*req_connect->release());
              });
}

void
impl::service::handle_connect_service(auto ch, auto args)
{
    using msg = fractos::service::compute::cuda::message::connect_service;

    //////////////////////////////////////////////////////////////////////
    // Check request correctness

    LOG_REQ("handle_connect_service");

    // If we cannot respond to it, ignore the request
    if (args->has_valid_cap(&msg::request::caps::cont, core::cap::request_tag) == 0) {
        LOG_RES("handle_connect_service")
            << " [error] request without continuation, ignoring";
        return;
    }

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.cont);

    // If it's malformed, respond appropriately
    if (not args->has_exactly_args()) {

        LOG_RES("handle_connect_service")
            << " error=ERR_OTHER";

        reqb_cont
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            // Use this macro to ignore invocation errors; i.e., don't fail
            // service if client revokes the continuation it passed to us
            .as_callback_log_ignore_error("error invoking continuation, ignoring");

        return;
    }

    //////////////////////////////////////////////////////////////////////
    // Process request

    LOG_RES("handle_connect_service")
        << " error=ERR_SUCCESS"
        << " make_device=" << _req_make_device.get_cid();

    // Send the service method requests as a response

    reqb_cont
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_cap(&msg::response::caps::make_device, _req_make_device)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("error invoking continuation, ignoring");
}

void
impl::service::handle_make_device(auto ch, auto args)
{
    using msg = fractos::service::compute::cuda::message::service::make_device;

    //////////////////////////////////////////////////////////////////////
    // Check request correctness

    // If we cannot respond to it, ignore the request
    if (args->has_valid_cap(&msg::request::caps::cont, core::cap::request_tag) == 0) {
        LOG_REQ("handle_make_device")
            << " [error] request without continuation, ignoring";
        return;
    }

    if (not args->has_exactly_args()) {

        LOG_REQ("handle_make_device")
            << " cont=" << args->caps.cont.get_cid()
            << " [error] malformed request";
        LOG_RES("handle_make_device")
            << " error=ERR_OTHER";

        ch->template make_request_builder<msg::response>(args->caps.cont)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_error("error invoking continuation, ignoring");

        return;
    }

    //////////////////////////////////////////////////////////////////////
    // Process request

    LOG_REQ("handle_make_device")
        << " value=" << args->imms.value.get()
        << " cont=" << args->caps.cont.get_cid();

    auto self = _self.lock();

    // Create new object

    auto obj = impl::make_device(self, args->imms.value);

    // Create request handlers for the object's methods
    obj->register_methods(ch)
        // Capture channel as it is used later. Also capture continuation
        // request, so args can be freed immediately, and put back to receive
        // future invocations
        .then([self, obj, ch, cont=std::move(args->caps.cont)](auto& fut) {
                  auto reqb_cont = ch->template make_request_builder<msg::response>(cont);

                  try {
                      fut.get();
                  } catch (...) {
                      LOG_RES_PTR("handle_make_device", self)
                          << " error=ERR_OTHER";
                      reqb_cont
                          .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
                          .on_channel()
                          .invoke()
                          .as_callback_log_ignore_error("error invoking continuation, ignoring");
                      return;
                  }

                  // Send requests back to client

                  LOG_RES_PTR("handle_make_device", self)
                      << " error=ERR_OTHER"
                      << " destroy=" << obj->req_destroy.get_cid();
                  ch->template make_request_builder<msg::response>(cont)
                      .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                      .set_cap(&msg::response::caps::destroy, obj->req_destroy)
                      .on_channel()
                      .invoke()
                      .as_callback_log_ignore_error("error invoking continuation, ignoring");

                  // Object cont is destroyed here, which implicitly calls
                  // cont.get_channel()->cap_close(cont). This is ok, since cont
                  // cont was received on the same channel we used to invoke the
                  // continuation with the handler's result, so both invoke() and
                  // cap_close() will be executed in strict order.
              })
        .as_callback();
}


impl::service::service(std::string name)
    :_requested_exit(false)
    ,_name(name)
{
}

void
impl::service::request_exit()
{
    _requested_exit.store(true);
}

bool
impl::service::exit_requested() const
{
    return _requested_exit.load();
}


std::shared_ptr<impl::service>
impl::make_service(std::string name)
{
    auto res = std::shared_ptr<impl::service>(new impl::service(name));
    res->_self = res;
    return res;
}

std::string
impl::to_string(const service& obj)
{
    return "service(" + obj._name + ")";
}
