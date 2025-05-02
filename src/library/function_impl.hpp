#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Function : public impl::Base<srv::Function, impl::Function> {
        Function(std::shared_ptr<fractos::core::channel> ch,
                 size_t args_total_size, std::vector<size_t> args_size,
                 fractos::wire::endian::uint8_t error,
                 fractos::core::cap::request req_func_call,
                 fractos::core::cap::request req_func_destroy);

        std::weak_ptr<Function> self;
        std::shared_ptr<fractos::core::channel> ch;
        size_t args_total_size;
        std::vector<size_t> args_size;

        fractos::wire::endian::uint8_t error;

        fractos::core::cap::request req_func_call;
        fractos::core::cap::request req_func_destroy;

        bool destroyed;
    };

    std::string to_string(const Function& obj);

}
