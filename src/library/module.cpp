
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include <module_impl.hpp>
#include <function_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

inline
Module_impl& Module_impl::get(Module& obj)
{
    return *reinterpret_cast<Module_impl*>(obj._pimpl.get());
}

inline
const Module_impl& Module_impl::get(const Module& obj) 
{
    return *reinterpret_cast<Module_impl*>(obj._pimpl.get());
}


Module::Module(std::shared_ptr<void> pimpl, std::string name) : _pimpl(pimpl) {

    DLOG(INFO) << "initialize module : " << name << " from file path";
}


// Module::Module(std::shared_ptr<void> pimpl, core::cap::memory contents, std::string name) : _pimpl(pimpl) {

//     DLOG(INFO) << "initialize module : " << name << " from data buffer";
// }

Module::~Module() {
    DLOG(INFO) << "Module: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

core::future<std::shared_ptr<Function>> Module::get_function(
            const std::string& func_name) {
    
        using msg = ::service::compute::cuda::wire::Module::get_function;
    
        DVLOG(logging::SERVICE) << "Module::get_function <-";
    
        auto& pimpl = Module_impl::get(*this);
    
        auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
        return pimpl.ch->make_request_builder<msg::request>(pimpl.req_get_func)
            // .set_imm(&msg::request::imms::name, name) // unsigned int vs uint32_t
            .set_imm(&msg::request::imms::func_name_size, func_name.size())
            .set_imm(offsetof(msg::request::imms, func_name), func_name.c_str(), func_name.size())
            .set_cap(&msg::request::caps::continuation, resp)
            .on_channel()
            .invoke(resp) // wait for srv_handle
            .unwrap()
            .then([func_name](auto& fut) { // function_name
                auto [ch, args] = fut.get();
    
                if (not args->has_exactly_args()) {
                    // throw core::other_error("invalvalue response format for Module::get_function");
                    DVLOG(logging::SERVICE) << "Context::get_function ->"
                    <<" error= OTHER args";
                }
    
                DVLOG(logging::SERVICE) << "Context::get_function ->"
                                        << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
                wire::error_raise_exception_maybe(args->imms.error);
    
                //get Function object
                std::shared_ptr<Function_impl> pimpl_(
                    new Function_impl{{}, ch, args->imms.error,
                            std::move(args->caps.call),
                            std::move(args->caps.func_destroy)}
                    );
                pimpl_->self = pimpl_;
                auto pimpl = static_pointer_cast<void>(pimpl_);
                std::shared_ptr<Function> res(new Function{pimpl, func_name});
                return res;
            });
    }

core::future<void> Module::destroy() {
    using msg = ::service::compute::cuda::wire::Module::destroy;

    DVLOG(logging::SERVICE) << "Module::destroy <-";

    auto& pimpl = Module_impl::get(*this);
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
                // throw core::other_error("invalid response format for Module::destroy");
                DVLOG(logging::SERVICE) << "Module::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Module::destroy ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}

