
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;

inline
cuda_context_impl& cuda_context_impl::get(cuda_context& obj)
{
    return *reinterpret_cast<cuda_context_impl*>(obj._pimpl.get());
}

inline
const cuda_context_impl& cuda_context_impl::get(const cuda_context& obj) 
{
    return *reinterpret_cast<cuda_context_impl*>(obj._pimpl.get());
}




cuda_context::cuda_context(std::shared_ptr<void> pimpl, wire::endian::uint8_t id) : _pimpl(pimpl) {
    CUcontext ctx;
    checkCudaErrors(cuCtxCreate(&ctx, CU_CTX_SCHED_SPIN, id));

    DLOG(INFO) << "initialize device : " << id;
}

cuda_context::cuda_context(std::shared_ptr<void> pimpl) : _pimpl(pimpl) {
}

cuda_context::cuda_context(wire::endian::uint8_t id) {}

cuda_context::~cuda_context() {
    DLOG(INFO) << "cuda_context: i am freed";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

// core::future<void> cuda_context::destroy() {
//     DLOG(INFO) << "cuda_context: destroy";
// }
core::future<void> cuda_context::destroy() {
    using msg = ::service::compute::cuda::message::cuda_context::destroy;

    DVLOG(logging::SERVICE) << "virtual_device::destroy <-";

    auto& pimpl = cuda_context_impl::get(*this);
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
                // throw core::other_error("invalid response format for cuda_context::destroy");
                DVLOG(logging::SERVICE) << "cuda_context::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "cuda_context::destroy ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

