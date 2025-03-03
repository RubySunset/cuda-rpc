
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




cuda_context::cuda_context(std::shared_ptr<void> pimpl, wire::endian::uint32_t value) : 
    _pimpl(pimpl) {


    DLOG(INFO) << "initialize context : " << value;
}

cuda_context::cuda_context(std::shared_ptr<void> pimpl) : _pimpl(pimpl) {
}

cuda_context::cuda_context(wire::endian::uint32_t id) {}

cuda_context::~cuda_context() {
    DLOG(INFO) << "cuda_context: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}



core::future<std::shared_ptr<cuda_memory>> cuda_context::make_cuda_Memalloc(
                    uint64_t size) {

    using msg = ::service::compute::cuda::message::cuda_context::make_cuda_Memalloc;

    LOG(INFO) << "cuda_context::make_cuda_Memalloc <-";

    auto& pimpl = cuda_context_impl::get(*this);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_cuda_Memalloc)
        .set_imm(&msg::request::imms::size, size) // unsigned int vs uint32_t
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([size](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalvalue response format for cuda_service::make_cuda_device");
                LOG(INFO) << "cuda_context::make_cuda_Memalloc ->"
                <<" error= OTHER args";
            }

            LOG(INFO) << "cuda_context::make_cuda_Memalloc ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);

            // get cuda_device object
            std::shared_ptr<cuda_memory_impl> pimpl_(
                new cuda_memory_impl{{}, ch, args->imms.error, 
                        std::move(args->caps.destroy)}
                );
            pimpl_->self = pimpl_;
            auto pimpl = static_pointer_cast<void>(pimpl_);
            std::shared_ptr<cuda_memory> res(new cuda_memory{pimpl, size});
            return res;
        });
}

// core::future<void> cuda_context::destroy() {
//     DLOG(INFO) << "cuda_context: destroy";
// }
core::future<void> cuda_context::destroy() {
    using msg = ::service::compute::cuda::message::cuda_context::destroy;

    DVLOG(logging::SERVICE) << "cuda_context::destroy <-";

    auto& pimpl = cuda_context_impl::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_ctx_destroy)
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

