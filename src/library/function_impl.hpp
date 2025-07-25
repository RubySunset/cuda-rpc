#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct FunctionState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Function> self;

        CUfunction cufunction;
        size_t args_total_size;
        std::vector<size_t> args_size;

        fractos::core::cap::request req_generic;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Function = fractos::common::service::ImplWrapper<clt::Function, impl::FunctionState>;

    std::shared_ptr<clt::Function>
    make_function(std::shared_ptr<fractos::core::channel> ch,
                  CUfunction cufunction,
                  size_t args_total_size, std::vector<size_t> args_size,
                  fractos::core::cap::request req_generic);

    std::string to_string(const Function& obj);
    std::string to_string(const FunctionState& obj);

}
