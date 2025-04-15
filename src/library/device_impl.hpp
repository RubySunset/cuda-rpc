#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Device {
        static Device& get(srv::Device& device);
        static const Device& get(const srv::Device& device);

        std::weak_ptr<Device> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::wire::endian::uint8_t error;
        fractos::core::cap::request req_make_context; // new
        // fractos::core::cap::request req_test;
        fractos::core::cap::request req_destroy;

        bool destroyed;
    };

}
