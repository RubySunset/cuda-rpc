
#include <utility>

// #include "./cuda_service.hpp"
#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>

#include <device_impl.hpp>
#include <context_impl.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
cuda_device_impl& cuda_device_impl::get(cuda_device& obj)
{
    return *reinterpret_cast<cuda_device_impl*>(obj._pimpl.get());
}

inline
const cuda_device_impl& cuda_device_impl::get(const cuda_device& obj) 
{
    return *reinterpret_cast<cuda_device_impl*>(obj._pimpl.get());
}

cuda_device::cuda_device(std::shared_ptr<void> pimpl, wire::endian::uint8_t id) : _pimpl(pimpl){ // value

    DLOG(INFO) << "initialize device : " << id;
}

cuda_device::cuda_device(std::shared_ptr<void> pimpl) : _pimpl(pimpl) {
}

cuda_device::cuda_device(wire::endian::uint8_t id) {}

cuda_device::~cuda_device() {
    DLOG(INFO) << "cuda_device: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

core::future<std::shared_ptr<cuda_context>> cuda_device::make_cuda_context(
                    unsigned int flags) {

    using msg = ::service::compute::cuda::message::cuda_device::make_cuda_context;

    DVLOG(logging::SERVICE) << "cuda_device::make_cuda_context <-";

    auto& pimpl = cuda_device_impl::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_make_cuda_context)
        .set_imm(&msg::request::imms::flags, flags) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([flags, this](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for cuda_service::make_cuda_device");
                DVLOG(logging::SERVICE) << "cuda_device::make_cuda_context ->"
                <<" error= OTHER args";
            }

            DVLOG(logging::SERVICE) << "cuda_device::make_cuda_context ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            // get cuda_device object
            std::shared_ptr<cuda_context_impl> pimpl_(
                new cuda_context_impl{{}, ch, args->imms.error, 
                        std::move(args->caps.make_cumemalloc),
                        std::move(args->caps.make_module_file),
                        std::move(args->caps.synchronize),
                        std::move(args->caps.destroy)}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<cuda_context> res(new cuda_context{pimpl, flags});
            return res;
        });
}

core::future<void> cuda_device::destroy() {
    using msg = ::service::compute::cuda::message::cuda_device::destroy;

    DVLOG(logging::SERVICE) << "virtual_device::destroy <-";

    auto& pimpl = cuda_device_impl::get(*this);
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
                // throw core::other_error("invalid response format for cuda_device::destroy");
                DVLOG(logging::SERVICE) << "cuda_device::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "cuda_device::destroy ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

