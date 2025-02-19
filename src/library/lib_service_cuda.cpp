#include <fractos/service/compute/cuda.hpp>
#include <fractos/wire/error.hpp>
#include <memory>

#include "./common.hpp"
#include "./lib_service_impl.hpp"
#include <cuda.h>

using namespace fractos;

/*
 * Here, we build the client-side representation of the remote example service
 * by retrieving the necessary capabilities, and wrapping them into an instance
 * of the example::service class.
 *
 * The class provides one method for each message type in the messages::service
 * namespace inside "src/include/fractos/service/example/message.hpp".
 */

core::future<std::unique_ptr<service::compute::cuda::Service>>
service::compute::cuda::make_service(std::shared_ptr<core::channel> ch,
                                fractos::core::gns::service& gns, const std::string& name,
                                const std::chrono::microseconds& wait_time)
{
    auto full_name = get_name(name);
 
    LOG_OP("compute::cuda::make_service")
         << " <-"
         << " name=" << full_name;
    return gns.get_wait_for<core::cap::request>(ch, full_name, wait_time)
        .then([ch, name](auto& fut) {
            core::cap::request req;
                try {
                    req = std::move(fut.get()); 
                    DLOG(INFO) << "Found service";
                } catch (const fractos::core::gns::token_error& e) {
                    LOG(INFO) << "Can't find service";
                }

                std::shared_ptr<impl::CuService_impl> pimpl_(
                    new impl::CuService_impl{{}, ch, std::move(req), name});
                pimpl_->self = pimpl_;
                auto pimpl = std::static_pointer_cast<void>(pimpl_);
                // unique_ptr<device_service> res(new device_service{pimpl});

                std::unique_ptr<service::compute::cuda::Service> res(new service::compute::cuda::Service{pimpl});
                // res->_pimpl = pimpl;

                
                return res;
              });

}


std::string
service::compute::cuda::to_string(const fractos::service::compute::cuda::Service& obj)
{
    auto& pimpl = impl::CuService_impl::get(obj);
    return impl::to_string(pimpl);
}


core::future<std::shared_ptr<service::compute::cuda::Device>>
service::compute::cuda::Service::make_cuda_device(std::shared_ptr<core::channel> ch, uint64_t value)
{
    LOG_REQ("make_cuda_device")
        << " value=" << value;

    using msg = service::compute::cuda::message::Service::make_cuda_device;

    auto& pimpl = impl::CuService_impl::get(*this);

    /*
     * 1) Create response_builder() to get the remote operation result.
     *
     * This is a shortcut to declaring a single-use handler on the client, that
     * the remote operation will invoke when it finishes, and returns the
     * response message contents.
     */
    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());

    /*
     * 2) Create and invoke remote operation.
     */
    auto fut = ch->make_request_builder<msg::request>(pimpl.req_make_cuda_device)
        .set_imm(&msg::request::imms::value, value)
        .set_cap(&msg::request::caps::cont, resp)
        .on_channel()
        .invoke(resp)
        .unwrap();

    /*
     * 3) Parse response message into method's result
     */

    return std::move(fut)
        .then([this](auto& fut) {
                auto [ch, args] = fut.get();
                wire::error_raise_exception_maybe(args->imms.error);

                // Build internal object detail::object
                std::shared_ptr<impl::CuDevice_impl> pimpl_(
                    new impl::CuDevice_impl{{}, ch,
                                            std::move(args->caps.destroy)});
                pimpl_->self = pimpl_;

                auto pimpl = std::static_pointer_cast<void>(pimpl_);
                // unique_ptr<device_service> res(new device_service{pimpl});

                std::unique_ptr<service::compute::cuda::Device> res(new service::compute::cuda::Device{pimpl});

                LOG_RES("make_device");

                // Return example::object
                return res;
              });
}

core::future<void>
service::compute::cuda::Device::destroy()
{
    auto& pimpl = impl::CuDevice_impl::get(*this);
    using msg = service::compute::cuda::message::Device::destroy;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());

    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_destroy)
        .set_cap(&msg::request::caps::cont, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([](auto& fut) {
                  (void)fut.get();
              });
}

service::compute::cuda::Device::~Device()
{
    // NOTE: Register future<void>::get() as a callback for whenever it is
    // ready. This is because we don't want to block the local destructor
    // waiting for the remote object destruction, and future objects must always
    // be waited for.
    destroy()
        .as_callback();
}

service::compute::cuda::Service::~Service()
{
    LOG_REQ("destroy service");
    // NOTE: Register future<void>::get() as a callback for whenever it is
    // ready. This is because we don't want to block the local destructor
    // waiting for the remote object destruction, and future objects must always
    // be waited for.
    CHECK(false);
}




 

// core::future<std::unique_ptr<service::compute::cuda::Service>>
// service::compute::cuda::make_service(std::shared_ptr<core::channel> ch,
//                                 fractos::core::gns::service& gns, const std::string& name,
//                                 const std::chrono::microseconds& wait_time)
//  {
//      auto full_name = get_name(name);
 
//      LOG_OP("compute::cuda::make_service")
//          << " <-"
//          << " name=" << full_name;
 
//      // Get the one request we need from the GNS, wait until it has been
//      // published for up to wait_time microseconds.
//      return gns.get_wait_for<core::cap::request>(ch, full_name, wait_time)
//          // -> future<cap::request>
//          .then([ch, full_name](auto& fut) {
//                    core::cap::request req;
//                    try {
//                        // future<cap::request> -> cap::request
//                        //   We need to move the returned request into `req`. This
//                        req = std::move(fut.get());
//                        DLOG(INFO) << "Found service";
//                    } catch (const fractos::core::gns::token_error& e) {
//                        // We couldn't find an entry with given name after all
//                        // this time, throw the relevant exception, so it will
//                        // propagate to the client when they wait for the future
//                        // to be resolved - e.g., example::make_service(...).get()
                       
//                        LOG(ERROR) << "Can't find service";
//                        std::stringstream ss;
//                        ss << "could not find service '" << full_name << "'";
//                        throw no_service_error(ss.str());
//                    }
//                    using msg = compute::cuda::message::connect_cuda_service;
 
//                    // Build an empty response message, that the server will fill
//                    // as a response to our invocation
//                    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
 
//                    // Use the request in GNS to connect to the service, which
//                    // returns all capabilities we need for the service class
//                    return ch->make_request_builder<msg::request>(req)
//                        .set_cap(&msg::request::caps::cont, resp)
//                        .on_channel(ch)
//                        .invoke(resp)
//                        // Keep connect_service cap around until creation of the
//                        // new with the selected args happens. This is a builder
//                        // graph (args contain other builders), so creation is not
//                        // immediate (args are created recursively).
//                        .then([req=std::move(req)](auto& fut) {
//                                  return fut.get();
//                              })
//                        .unwrap(); // future<future<...>> -> future<...>
//                })  // -> future<future<...>>
//          .unwrap() // ->               future<...>
//          // Response handler, which passes the channel where the response was
//          // received, and the core::receive_args_ptr with the contents of the
//          // response invocation
//          .then([name](auto& fut) {
//                    auto [ch, args] = fut.get();
 
//                    // Throw exception iff args->imms.error != ERR_SUCCESS. The
//                    // exception will be used to fulfil the promise via
//                    // promise::set_exception().
//                    wire::error_raise_exception_maybe(args->imms.error);
 
//                    // Build service PIMPL object, with the requests sent by the
//                    // server
//                    std::shared_ptr<impl::CuService_impl> pimpl(
//                        new impl::CuService_impl{{}, ch, std::move(args->caps.make_device), name});
//                    pimpl->self = pimpl;
 
//                    std::unique_ptr<service::compute::cuda::Service> res(new service::compute::cuda::Service);
//                    res->_pimpl = pimpl;
 
//                    LOG_OP("example::make_service")
//                        << " ->"
//                        << " " << to_string(*res);
 
//                    // Return example::service object to the client
//                    return res;
//                }); // -> future<std::unique_ptr<example::service>>
// }



/****************************************************************************
 *
 * CUdevice
 */
// service::compute::cuda::Device::~Device()
// {
//     auto& pimpl = impl::Device::get(*this); // 
//     pimpl.destroy()
//         .get();
// }

// core::future<std::shared_ptr<service::compute::cuda::Device>>
// service::compute::cuda::Device::make_context(const std::vector<CUctxCreateParams>& params,  unsigned int flags)
// {
//     auto& pimpl = impl::Device::get(*this); // 
//     CHECK(false);
// }

// core::future<void>
// service::compute::cuda::Device::destroy()
// {
//     auto& pimpl = impl::Device::get(*this); // 
//     if (not pimpl.destroy_sent.test_and_set()) {
//         CHECK(false);
//     }
// }




/****************************************************************************
 *
 * CUcontext
//  */

// Context::~Context()
// {
//     //
// }

// core::future<std::shared_ptr<Context>>
// Context::make_module(uint64_t device_id)
// {
//     //
// }

// core::future<void>
// Context::destroy()
// {
//     //
// }
