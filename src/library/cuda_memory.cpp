
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;

inline
cuda_memory_impl& cuda_memory_impl::get(cuda_memory& obj)
{
    return *reinterpret_cast<cuda_memory_impl*>(obj._pimpl.get());
}

inline
const cuda_memory_impl& cuda_memory_impl::get(const cuda_memory& obj) 
{
    return *reinterpret_cast<cuda_memory_impl*>(obj._pimpl.get());
}




cuda_memory::cuda_memory(std::shared_ptr<void> pimpl, wire::endian::uint64_t size) : _pimpl(pimpl) {



    DLOG(INFO) << "initialize memory : " << size;
}

cuda_memory::cuda_memory(std::shared_ptr<void> pimpl) : _pimpl(pimpl) {
}

cuda_memory::cuda_memory(wire::endian::uint64_t size) {}

cuda_memory::~cuda_memory() {
    DLOG(INFO) << "cuda_memory: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

// core::future<void> cuda_memory::destroy() {
//     DLOG(INFO) << "cuda_memory: destroy";
// }
core::future<void> cuda_memory::destroy() {
    using msg = ::service::compute::cuda::message::cuda_memory::destroy;

    DVLOG(logging::SERVICE) << "cuda_memory::destroy <-";

    auto& pimpl = cuda_memory_impl::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_mem_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for cuda_memory::destroy");
                DVLOG(logging::SERVICE) << "cuda_memory::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "cuda_memory::destroy ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

