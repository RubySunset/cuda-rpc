#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct ModuleState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Module> self;
        fractos::core::cap::request req_generic;
        fractos::core::cap::request req_get_func;
        fractos::core::cap::request req_module_unload;

        CUmodule cumodule;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Module = fractos::common::service::ImplWrapper<clt::Module, impl::ModuleState>;

    std::shared_ptr<clt::Module>
    make_module(std::shared_ptr<fractos::core::channel> ch,
                CUmodule cumodule,
                fractos::core::cap::request req_generic,
                fractos::core::cap::request req_get_func,
                fractos::core::cap::request req_module_unload);

    std::string to_string(const Module& obj);
    std::string to_string(const ModuleState& obj);
}
