#include <utility>

// #include "./Service.hpp"
#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include <service_impl.hpp>
#include <device_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace std;

using namespace impl;
using namespace fractos::service::compute::cuda;
///////////////////////////////////////////////////////////
// * pimpl pattern
// * all Service_impl related functions
inline
Service_impl& Service_impl::get(Service& obj)
{
    return *reinterpret_cast<Service_impl*>(obj._pimpl.get());
}

inline
const Service_impl& Service_impl::get(const Service& obj)
{
    return *reinterpret_cast<Service_impl*>(obj._pimpl.get());
}

Service::Service(std::shared_ptr<void> pimpl) : _pimpl(pimpl) {
    
}


std::shared_ptr<core::channel> Service::get_default_channel() {
    auto& pimpl = Service_impl::get(*this);
    return pimpl.ch;
}

std::shared_ptr<core::channel> Service::get_default_channel() const {
    auto& pimpl = Service_impl::get(*this);
    return pimpl.ch;
}

void Service::set_default_channel(std::shared_ptr<core::channel> ch) {
    auto& pimpl = Service_impl::get(*this);
    pimpl.ch = ch;
}
/////////////////////////////////////////////////////////

/*
 * Make a Service from global_ns which the server publishes to 
 */
core::future<std::unique_ptr<Service>>
fractos::service::compute::cuda::make_service(fractos::core::gns::service& gns, const std::string& name,
                                    std::shared_ptr<core::channel> ch) {
    
    return gns.get_wait_for<core::cap::request>(ch, name, std::chrono::seconds{0})
        .then([ch, name](auto& fut) {
            core::cap::request req;
                try {
                    req = std::move(fut.get()); 
                    DLOG(INFO) << "Found service";
                } catch (const fractos::core::gns::token_error& e) {
                    LOG(INFO) << "Can't find service";
                }

                std::shared_ptr<Service_impl> pimpl_(
                    new Service_impl{{}, ch, std::move(req)});
                pimpl_->self = pimpl_;
                auto pimpl = std::static_pointer_cast<void>(pimpl_);
                std::unique_ptr<Service> res(new Service{pimpl});

                return res;
              });

} 


/*
 *  Make Device frontend function
 *  makes a request for make_device and sets the continuation of the response
 */
core::future<std::shared_ptr<Device>> Service::make_device(uint8_t value) {
    using msg = ::service::compute::cuda::wire::Service::make_device;

    DVLOG(logging::SERVICE) << "Service::make_device <-";

    auto& pimpl = Service_impl::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_make_device)
        .set_imm(&msg::request::imms::value, value)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([value](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for Service::make_device");
                DVLOG(logging::SERVICE) << "Service::make_device ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Service::make_device ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            // get Device object
            std::shared_ptr<Device_impl> pimpl_(
                new Device_impl{{}, ch, args->imms.error, 
                        std::move(args->caps.make_context),
                        std::move(args->caps.destroy)}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Device> res(new Device{pimpl, value});
            return res;
        });
}


/*
 *  Get published Device
 */
core::future<std::shared_ptr<Device>> Service::get_Device(fractos::core::gns::service& gns, 
                                                                               uint8_t value) {
    using msg = ::service::compute::cuda::wire::Service::get_Device;

    DVLOG(logging::SERVICE) << "Service::get_Device <-";

    auto& pimpl = Service_impl::get(*this);

    const string name = "get_vdev";
    auto ch = pimpl.ch;

    return gns.get_wait_for<core::cap::request>(ch, name, chrono::seconds{0})
        .then([ch, name, value](auto& fut) {
            core::cap::request get_vdev = std::move(fut.get());

                auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
                return ch->make_request_builder<msg::request>(get_vdev)
                    .set_imm(&msg::request::imms::value, value)
                    .set_cap(&msg::request::caps::continuation, resp)
                    .on_channel()
                    .invoke(resp)
                    .unwrap()
                    .then([ch, value](auto& fut) {
                        auto [ch, args] = fut.get();

                        if (not args->has_exactly_args()) {
                            // throw core::other_error("invalvalue response format for Service::get_Device");
                            DVLOG(logging::SERVICE) << "Service::get_Device ->"
                                                << " error=OTHER args" ;
                        }

                        DVLOG(logging::SERVICE) << "Service::get_Device ->"
                                                << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
                        wire::error_raise_exception_maybe(args->imms.error);

                        shared_ptr<Device_impl> pimpl_ (
                            new Device_impl{{}, ch, args->imms.error, 
                                    // move(args->caps.allocate_memory),
                                    // move(args->caps.register_function),
                                    move(args->caps.make_context),
                                    move(args->caps.destroy)}
                            ); 
                        pimpl_->self = pimpl_;
                        auto pimpl = static_pointer_cast<void>(pimpl_);
                        shared_ptr<Device> res(new Device{pimpl, value});
                        return res;
                    });
        })
        .unwrap();
}

