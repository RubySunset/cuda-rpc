// #pragma once
#include <fractos/service/compute/cuda.hpp>


#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <./src_service.hpp>
#include <./common.hpp>


using namespace fractos;
using namespace impl;
// using namespace fractos::service::compute::cuda;


std::string
impl::to_string(const service& obj)
{
    return "service(" + obj._name + ")";
}

std::string
impl::to_string(const device& obj)
{
    std::stringstream ss;
    ss << "object(" << &obj << ")";
    return ss.str();

}


impl::service::service(std::string name)
    :_requested_exit(false)
    ,_name(name)
{
}


impl::device::device(std::shared_ptr<service> srv, uint64_t value)
:value(value)
{
}

//////////////////////////////////////////////////
// service
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
impl::make_cuda_service(std::string name)
{
    auto res = std::shared_ptr<impl::service>(new impl::service(name));
    res->_self = res;
    return res;
}


core::future<core::cap::request>
impl::service::register_methods(std::shared_ptr<core::channel> ch)
{
    LOG_REQ("register_methods");
    auto self = _self.lock();
    auto reqb_connect_cuda_service =
        ch->make_request_builder<fractos::service::compute::cuda::message::connect_cuda_service::request>(
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
            // fractos::service::example::message::connect_service::request::imms,
            // and args->caps has type
            // fractos::service::example::message::connect_service::request::caps.
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
                LOG(INFO) << "In register_method handler";
                self->handle_connect_cuda_service(ch, std::move(args));
            });

    auto reqb_make_cuda_device =
        ch->make_request_builder<fractos::service::compute::cuda::message::Service::make_cuda_device::request>(
            ch->get_default_endpoint(),
            [self](auto ch, auto args) {
                self->handle_make_device(ch, std::move(args));
            });

                
    auto req_connect = std::make_shared<std::unique_ptr<core::cap::request>>();

    return reqb_connect_cuda_service
        .on_channel()
        .make_request()
        // Return the connect_service request after all else is ready
        .then([req_connect, reqb_make_cuda_device, self](auto& fut) {
            req_connect->reset(new core::cap::request(fut.get()));
            return reqb_make_cuda_device
                .on_channel()
                .make_request()
                .then([self](auto& fut) {
                          self->_req_make_cuda_device = fut.get();
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
impl::service::handle_connect_cuda_service(auto ch, auto args)
{
    LOG_REQ("In connect_cuda_service handler");
    using msg = fractos::service::compute::cuda::message::connect_cuda_service;

    //////////////////////////////////////////////////////////////////////
    // Check request correctness
    // LOG("handle_connect_cuda_service");
    LOG_REQ("handle_connect_cuda_service");

    // If we cannot respond to it, ignore the request
    if (args->has_valid_cap(&msg::request::caps::cont, core::cap::request_tag) == 0) {
        LOG_RES("handle_connect_cuda_service")
            << " [error] request without continuation, ignoring";
        return;
    }

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.cont);

    // If it's malformed, respond appropriately
    if (not args->has_exactly_args()) {

        LOG_RES("handle_connect_cuda_service")
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

    LOG_RES("handle_connect_cuda_service")
        << " error=ERR_SUCCESS"
        << " connect_cuda_service=" << _req_make_cuda_device.get_cid();

    // Send the service method requests as a response

    reqb_cont
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_cap(&msg::response::caps::make_cuda_device, _req_make_cuda_device)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("error invoking continuation, ignoring");
}


void
impl::service::handle_make_device(auto ch, auto args)
{
    LOG_REQ("In make_device handler");
    using msg = fractos::service::compute::cuda::message::Service::make_cuda_device;
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

    LOG_REQ("handle_make_device")
        << " value=" << args->imms.value.get()
        << " cont=" << args->caps.cont.get_cid();

    auto self = _self.lock();

    // Create new object

    auto obj = impl::make_device(self, args->imms.value);

    // Create request handlers for the object's methods
    obj->register_device_methods(ch)
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
                      << " context=" << obj->req_make_context.get_cid()
                      << " destroy=" << obj->req_destroy.get_cid();
                  ch->template make_request_builder<msg::response>(cont)
                      .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
                    //   .set_cap(&msg::response::caps::make_cuda_context, obj->req_make_context)
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

//////////////////////////////////////////////////
// device

std::shared_ptr<impl::device>
impl::make_device(std::shared_ptr<impl::service> srv, uint64_t value)
{
    auto res = std::shared_ptr<impl::device>(new impl::device(srv, value));
    res->self = res;
    return res;
}

fractos::core::future<void>
impl::device::register_device_methods(std::shared_ptr<fractos::core::channel> ch)
{
    namespace msg = fractos::service::compute::cuda::message::Device;

    auto self = this->self.lock();

    auto reqb_make_context = ch->make_request_builder<msg::make_cuda_context::request>(
        ch->get_default_endpoint(),
        [self](auto ch, auto args) {
            self->handle_make_context(ch, std::move(args));
        });

    auto reqb_destroy = ch->make_request_builder<msg::destroy::request>(
            ch->get_default_endpoint(),
            [self](auto ch, auto args) {
                self->handle_destroy(ch, std::move(args));
            });
    


    return reqb_make_context.on_channel().make_request()
        .then([self, reqb_destroy](auto& fut) {
                  self->req_make_context = fut.get();
                  return reqb_destroy.on_channel().make_request();
              })
        .unwrap()
        .then([self](auto& fut) {
                  self->req_destroy = fut.get();
              });

}


void
impl::device::handle_make_context(auto ch, auto args) // unimplemented
{
    using msg = fractos::service::compute::cuda::message::Device::make_cuda_context;

    if (not args->has_valid_cap(&msg::request::caps::cont, core::cap::request_tag)) {
        LOG_REQ("handle_fetch")
            << " [error] request without continuation, ignoring";
        return;
    }

    auto reqb_cont = ch->template make_request_builder<msg::response>(args->caps.cont);

    if (not args->has_exactly_args()) {

        LOG_REQ("handle_make_context")
            << " cont=" << args->caps.cont.get_cid()
            << " [error] malformed request";
        LOG_RES("handle_make_context")
            << " error=ERR_OTHER";

        ch->template make_request_builder<msg::response>(args->caps.cont)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_error("error invoking continuation, ignoring");

        return;
    }

    LOG_REQ("handle_make_context")
        << " cont=" << args->caps.cont.get_cid();

    // auto res = value.load();

    LOG_RES("handle_make_context")
        << " error=ERR_SUCCESS";
        // << " num=" << res;

    reqb_cont
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        // .set_imm(&msg::response::imms::num, res)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("error invoking continuation, ignoring");
}




void
impl::device::handle_destroy(auto ch, auto args)
{
    /*
     * Revoke all capabilities, which effectively invalidates the object for all
     * remote clients; they cannot invoke any requests for the methods of this
     * object instance, while other objects still work as usual (each object has
     * independent method handler requests).
     *
     * After a method request is revoked, the handler function is destroyed. At
     * this point, the shared object pointer each handler holds is released, and
     * the object automatically deleted after the last of the handlers is gone.
     *
     * This approach ties method handling capabilities to the server-side
     * object, using the RAII pattern.
     */

    using msg = fractos::service::compute::cuda::message::Device::destroy;

    if (not args->has_valid_cap(&msg::request::caps::cont, core::cap::request_tag)) {
        LOG_REQ("handle_destroy")
            << " [error] request without continuation, ignoring";
        return;
    }

    if (not args->has_exactly_args()) {

        LOG_REQ("handle_destroy")
            << " cont=" << args->caps.cont.get_cid()
            << " [error] malformed request";
        LOG_RES("handle_destroy")
            << " error=ERR_OTHER";

        ch->template make_request_builder<msg::response>(args->caps.cont)
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)
            .on_channel()
            .invoke()
            .as_callback_log_ignore_error("error invoking continuation, ignoring");

        return;
    }

    LOG_REQ("handle_destroy")
        << " cont=" << args->caps.cont.get_cid();

    // revoke each of the methods in turn, then invoke the response continuation

    auto self = this->self.lock();

    ch->revoke(self->req_destroy) // add
        .then([ch, self, args=std::move(args)](auto& fut) {
                  auto error = wire::ERR_SUCCESS;
                  try {
                      fut.get();
                  } catch (...) {
                      error = wire::exception_to_error(std::current_exception());
                  }

                  LOG_RES_PTR("handle_destroy", self)
                      << " error=" << wire::to_string(error);

                  ch->template make_request_builder<msg::response>(args->caps.cont)
                      .set_imm(&msg::response::imms::error, error)
                      .on_channel()
                      .invoke()
                      .as_callback_log_ignore_error("error invoking continuation, ignoring");
              })
        .as_callback();
}












// core::future<std::shared_ptr<Service>>
// service::compute::cuda::make_service(std::shared_ptr<core::channel> ch,
//              core::gns::service& gns, const std::string& name,
//              const std::chrono::microseconds& wait_time = std::chrono::seconds{0})
// {
//     auto full_name = get_name(name);

//     LOG_OP("cuda::make_service")
//         << " <-"
//         << " name=" << full_name;

//     // Get the one request we need from the GNS, wait until it has been
//     // published for up to wait_time microseconds.

//     return gns.get_wait_for<core::cap::request>(ch, name, chrono::seconds{0})
//         .then([ch, name](auto& fut) {
//             core::cap::request req;
//                 try {
//                     req = std::move(fut.get()); 
//                     DLOG(INFO) << "Found service";
//                 } catch (const fractos::core::gns::token_error& e) {
//                     LOG(INFO) << "Can't find service";
//                 }

//                 // Build service PIMPL object, with the requests sent by the
//                 // server
//                 std::shared_ptr<service> pimpl_(
//                     new service{{}, ch, std::move(req)});
//                 pimpl_->self = pimpl_;
//                 auto pimpl = std::static_pointer_cast<void>(pimpl_);
//                 unique_ptr<Service> res(new Service{pimpl});
//                 res->_pimpl = pimpl; // ? depulcate>

//                 LOG_OP("example::make_service_cuda")
//                     << " ->"
//                     << " " << to_string(*res);

//                 // Return Service object to client.
                
//                 return res;
//             });
// }

// core::future<std::shared_ptr<Service>>
// make_service(std::shared_ptr<core::channel> ch,
//              core::cap::request& service_req)
// {
//     CHECK(false);
// }


// fractos::service::compute::cuda::Service::Service()
//     :_pimpl(service::make_service(std::string name))
// {
// }


// fractos::service::compute::cuda::Service::Service(std::string name)
//     :_pimpl(service::make_service(std::string name))
// {
// }







// fractos::service::compute::cuda::Service::Service()
// {
// }

// fractos::service::compute::cuda::Service::~Service()
// {
//     auto& pimpl = service::get(*this);
//     CHECK(false);
// }





// core::future<std::shared_ptr<Device>>
// Service::make_device(uint64_t device_id)
// {
//     auto& pimpl = service::get(*this);
//     CHECK(false);
// }

// core::future<void>
// Service::destroy()
// {
//     auto& pimpl = service::get(*this);
//     if (not pimpl.destroy_sent.test_and_set()) {
//         CHECK(false);
//     }
// }
