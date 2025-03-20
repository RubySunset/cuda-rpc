
#include <utility>

// #include "./Service.hpp"
#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>

#include <device_impl.hpp>
#include <context_impl.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
Device_impl& Device_impl::get(Device& obj)
{
    return *reinterpret_cast<Device_impl*>(obj._pimpl.get());
}

inline
const Device_impl& Device_impl::get(const Device& obj) 
{
    return *reinterpret_cast<Device_impl*>(obj._pimpl.get());
}

Device::Device(std::shared_ptr<void> pimpl, wire::endian::uint8_t id) : _pimpl(pimpl){ // value

    DLOG(INFO) << "initialize device : " << id;
}


Device::~Device() {
    DLOG(INFO) << "Device: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

core::future<std::shared_ptr<Context>> Device::make_context(
                    unsigned int flags) {

    using msg = ::service::compute::cuda::wire::Device::make_context;

    DVLOG(logging::SERVICE) << "Device::make_context <-";

    auto& pimpl = Device_impl::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_make_context)
        .set_imm(&msg::request::imms::flags, flags) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([flags, this](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for Device::make_context");
                DVLOG(logging::SERVICE) << "Device::make_context ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "Device::make_context ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            // get Device object
            std::shared_ptr<Context_impl> pimpl_(
                new Context_impl{{}, ch, args->imms.error, 
                        std::move(args->caps.make_memory),
                        std::move(args->caps.make_stream),
                        // std::move(args->caps.make_module_data),
                        std::move(args->caps.make_module_file),
                        std::move(args->caps.synchronize),
                        std::move(args->caps.destroy)}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<Context> res(new Context{pimpl, flags});
            return res;
        });
}

core::future<void> Device::destroy() {
    using msg = ::service::compute::cuda::wire::Device::destroy;

    DVLOG(logging::SERVICE) << "virtual_device::destroy <-";

    auto& pimpl = Device_impl::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Device::destroy");
                DVLOG(logging::SERVICE) << "Device::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Device::destroy ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

