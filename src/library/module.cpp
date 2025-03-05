
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <module_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
cuda_module_impl& cuda_module_impl::get(cuda_module& obj)
{
    return *reinterpret_cast<cuda_module_impl*>(obj._pimpl.get());
}

inline
const cuda_module_impl& cuda_module_impl::get(const cuda_module& obj) 
{
    return *reinterpret_cast<cuda_module_impl*>(obj._pimpl.get());
}




cuda_module::cuda_module(std::shared_ptr<void> pimpl, std::string name) : _pimpl(pimpl) {



    DLOG(INFO) << "initialize module : " << name;
}

cuda_module::cuda_module(std::shared_ptr<void> pimpl) : _pimpl(pimpl) {
}

cuda_module::cuda_module(std::string name) {}

cuda_module::~cuda_module() {
    DLOG(INFO) << "cuda_module: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

// core::future<void> cuda_module::destroy() {
//     DLOG(INFO) << "cuda_module: destroy";
// }
core::future<void> cuda_module::destroy() {
    using msg = ::service::compute::cuda::message::cuda_module::destroy;

    DVLOG(logging::SERVICE) << "cuda_module::destroy <-";

    auto& pimpl = cuda_module_impl::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_module_unload)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for cuda_module::destroy");
                DVLOG(logging::SERVICE) << "cuda_module::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "cuda_module::destroy ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

