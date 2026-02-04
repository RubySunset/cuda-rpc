#pragma once

#include "fractos/service/compute/cuda.hpp"
#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct CublasHandleState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::CublasHandle> self;

        std::weak_ptr<clt::Context> ctx;
        fractos::core::cap::request req_generic;
        cublasHandle_t handle;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using CublasHandle = fractos::common::service::ImplWrapper<clt::CublasHandle, impl::CublasHandleState>;

    std::shared_ptr<clt::CublasHandle>
    make_cublas_handle(clt::Context& ctx,
                std::shared_ptr<fractos::core::channel> ch,
                cublasHandle_t handle,
                fractos::core::cap::request req_generic);

    std::string to_string(const CublasHandle& obj);
    std::string to_string(const CublasHandleState& obj);

}
