#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>

#include <./common.hpp>
#include <service_impl.hpp>
#include <device_impl.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


std::string
srv::to_string(const srv::Service& obj)
{
    auto& pimpl = impl::Service::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Service& obj)
{
    std::stringstream ss;
    ss << "cuda::Service(" << &obj << ")";
    return ss.str();
}

inline
impl::Service&
impl::Service::get(srv::Service& obj)
{
    return *reinterpret_cast<impl::Service*>(obj._pimpl.get());
}

inline
const impl::Service&
impl::Service::get(const srv::Service& obj)
{
    return *reinterpret_cast<impl::Service*>(obj._pimpl.get());
}


impl::Service::Service(std::shared_ptr<fractos::core::channel> ch,
                       fractos::core::cap::request req_connect,
                       fractos::core::cap::request req_driver_get_version,
                       fractos::core::cap::request req_make_device)
    :ch(ch)
    ,req_connect(std::move(req_connect))
    ,req_driver_get_version(std::move(req_driver_get_version))
    ,req_make_device(std::move(req_make_device))
{
}

srv::Service::Service(std::shared_ptr<void> pimpl)
    : _pimpl(pimpl)
{
}


std::shared_ptr<core::channel>
srv::Service::get_default_channel()
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.ch;
}

std::shared_ptr<core::channel>
srv::Service::get_default_channel() const
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.ch;
}

void
srv::Service::set_default_channel(std::shared_ptr<core::channel> ch)
{
    auto& pimpl = impl::Service::get(*this);
    pimpl.ch = ch;
}


core::future<std::unique_ptr<srv::Service>>
srv::make_service(std::shared_ptr<core::channel> ch,
                  core::gns::service& gns, const std::string& name,
                  const std::chrono::microseconds& wait_time)
{
    static const std::string method = "service::compute::cuda::make_service[gns]";

    LOG_OP(method)
        << " name=" << name
        << " <- {}";

    return gns.get_wait_for<core::cap::request>(ch, name, wait_time)
        .then([ch, name](auto& fut) {
            auto req = fut.get();
            LOG_OP(method)
                << " -> " << core::to_string(req);
            return make_service(ch, req)
                .then([req=std::move(req)](auto& fut) {
                    return fut.get();
                });
        })
        .unwrap();
}

core::future<std::unique_ptr<srv::Service>>
srv::make_service(std::shared_ptr<core::channel> ch,
                  const core::cap::request& connect)
{
    static const std::string method = "service::compute::cuda::make_service[connect]";
    using msg = srv::wire::Service::connect;

    LOG_OP(method)
        << " connect=" << core::to_string(connect)
        << " <- {}";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(connect)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_OP(method)
                << " -> " << wire::to_string(*args);

            if (not args->has_exactly_args()) {
                throw core::other_error(method + ": invalid response format");
            }
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            auto pimpl_ = std::make_shared<impl::Service>(
                ch,
                std::move(args->caps.connect),
                std::move(args->caps.get_driver_version),
                std::move(args->caps.make_device));
            pimpl_->self = pimpl_;
            auto pimpl = std::static_pointer_cast<void>(pimpl_);
            std::unique_ptr<Service> res(new Service{pimpl});

            return res;
        });
}

const core::cap::request&
srv::Service::get_connect() const
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.req_connect;
}


/*
 *  Make Device frontend function
 *  makes a request for make_device and sets the continuation of the response
 */
core::future<std::shared_ptr<srv::Device>>
srv::Service::make_device(uint8_t value)
{
    using msg = ::service::compute::cuda::wire::Service::make_device;

    DVLOG(logging::SERVICE) << "Service::make_device <-";

    auto& pimpl = impl::Service::get(*this);

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
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            // get Device object
            std::shared_ptr<impl::Device> pimpl_(
                new impl::Device{{}, ch, args->imms.error, 
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
core::future<std::shared_ptr<srv::Device>>
srv::Service::get_Device(fractos::core::gns::service& gns, uint8_t value)
{
    using msg = ::service::compute::cuda::wire::Service::get_Device;

    DVLOG(logging::SERVICE) << "Service::get_Device <-";

    auto& pimpl = impl::Service::get(*this);

    const std::string name = "get_vdev";
    auto ch = pimpl.ch;

    return gns.get_wait_for<core::cap::request>(ch, name, std::chrono::seconds{0})
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
                                                << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
                        fractos::wire::error_raise_exception_maybe(args->imms.error);

                        std::shared_ptr<impl::Device> pimpl_ (
                            new impl::Device{{}, ch, args->imms.error, 
                                    // move(args->caps.allocate_memory),
                                    // move(args->caps.register_function),
                                    std::move(args->caps.make_context),
                                    std::move(args->caps.destroy)}
                            ); 
                        pimpl_->self = pimpl_;
                        auto pimpl = static_pointer_cast<void>(pimpl_);
                        std::shared_ptr<Device> res(new Device{pimpl, value});
                        return res;
                    });
        })
        .unwrap();
}
