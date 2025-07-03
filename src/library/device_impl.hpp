#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct DeviceState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Device> self;
        CUdevice device;
        fractos::core::cap::request req_generic;
        fractos::core::cap::request req_destroy;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Device = fractos::common::service::ImplWrapper<clt::Device, impl::DeviceState>;

    std::shared_ptr<clt::Device>
    make_device(std::shared_ptr<fractos::core::channel> channel,
                CUdevice device,
                fractos::core::cap::request req_generic,
                fractos::core::cap::request req_destroy);

    std::string to_string(const Device& obj);
    std::string to_string(const DeviceState& obj);

}
