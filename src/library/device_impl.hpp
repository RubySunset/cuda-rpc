#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Device : public impl::Base<srv::Device, impl::Device> {
        Device(std::shared_ptr<fractos::core::channel> channel,
               CUdevice device,
               fractos::core::cap::request req_generic,
               fractos::core::cap::request req_make_context,
               fractos::core::cap::request req_destroy);

        std::weak_ptr<srv::Device> self;
        std::shared_ptr<fractos::core::channel> ch;

        const CUdevice device;
        fractos::core::cap::request req_generic;
        fractos::core::cap::request req_make_context; // new
        // fractos::core::cap::request req_test;
        fractos::core::cap::request req_destroy;

        fractos::core::future<void> do_destroy();
    };

    std::string to_string(const Device& obj);

}
