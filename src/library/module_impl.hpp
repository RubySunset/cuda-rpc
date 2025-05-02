#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Module {
        static Module& get(srv::Module& module);
        static const Module& get(const srv::Module& module);
        Module(std::shared_ptr<fractos::core::channel> ch,
               fractos::wire::endian::uint8_t error,
               fractos::core::cap::request req_generic,
               fractos::core::cap::request req_get_func,
               fractos::core::cap::request req_module_unload);

        std::weak_ptr<Module> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::wire::endian::uint8_t error;
        fractos::core::cap::request req_generic;
        fractos::core::cap::request req_get_func;
        fractos::core::cap::request req_module_unload;

        bool destroyed;
    };

    std::string to_string(const Module& obj);
}
