#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Device {
        static Device& get(srv::Device& device);
        static const Device& get(const srv::Device& device);

        Device(std::shared_ptr<fractos::core::channel> channel,
               CUdevice device,
               fractos::core::cap::request req_make_context,
               fractos::core::cap::request req_destroy);

        std::weak_ptr<Device> self;
        std::shared_ptr<fractos::core::channel> ch;

        const CUdevice device;
        fractos::core::cap::request req_make_context; // new
        // fractos::core::cap::request req_test;
        fractos::core::cap::request req_destroy;

        bool destroyed;
    };

}
