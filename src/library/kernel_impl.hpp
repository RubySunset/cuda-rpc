#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

#include "./common.hpp"


namespace impl {

    namespace clt = ::fractos::service::compute::cuda;

    struct KernelState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Kernel> self;
        CUkernel cukernel;

        fractos::core::cap::request req_generic;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Kernel = fractos::common::service::ImplWrapper<clt::Kernel, impl::KernelState>;

    std::shared_ptr<clt::Kernel>
    make_kernel(std::shared_ptr<fractos::core::channel> ch,
                CUkernel cukernel,
                fractos::core::cap::request req_generic);

    std::string to_string(const Kernel& obj);
    std::string to_string(const KernelState& obj);
}
