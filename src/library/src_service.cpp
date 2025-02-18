// #pragma once
#include <fractos/service/compute/cuda.hpp>


#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <./src_service.hpp>
#include <./common.hpp>


using namespace fractos;
// using namespace fractos::service::compute::cuda;


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
impl::make_cuda_service(std::string name)
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



core::future<core::cap::request>
impl::service::register_methods(std::shared_ptr<core::channel> ch)
{
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
                self->handle_connect_cuda_service(ch, std::move(args));
            });
    auto req_connect = std::make_shared<std::unique_ptr<core::cap::request>>();

    return reqb_connect_cuda_service
        .on_channel()
        .make_request()
        // Return the connect_service request after all else is ready
        .then([req_connect](auto& fut) {
                  fut.get();
                  return std::move(*req_connect->release());
              });


    CHECK(false);
}

void
impl::service::handle_connect_cuda_service(auto ch, auto args)
{
    using msg = fractos::service::compute::cuda::message::connect_cuda_service;

    //////////////////////////////////////////////////////////////////////
    // Check request correctness

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
        << " make_object=" << _req_make_device.get_cid();

    // Send the service method requests as a response

    reqb_cont
        .set_imm(&msg::response::imms::error, wire::ERR_SUCCESS)
        .set_cap(&msg::response::caps::make_device, _req_make_device)
        .on_channel()
        .invoke()
        .as_callback_log_ignore_error("error invoking continuation, ignoring");
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
