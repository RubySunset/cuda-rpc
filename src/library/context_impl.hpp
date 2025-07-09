#pragma once

#include <cuda.h>
#include <fractos/common/service/impl_base.hpp>
#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/future.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct ContextState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Context> self;

        CUcontext context;
        std::weak_ptr<clt::Device> device;

        fractos::core::cap::request req_generic;
        fractos::core::cap::request req_module_data; // file
        fractos::core::cap::request req_ctx_sync;
        fractos::core::cap::request req_ctx_destroy;

        // an opaque data structure in libcuda
        std::unique_ptr<char[]> context_ptr;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Context = fractos::common::service::ImplWrapper<clt::Context, impl::ContextState>;

    std::shared_ptr<clt::Context>
    make_context(std::shared_ptr<fractos::core::channel> ch,
                 std::shared_ptr<clt::Device> device,
                 fractos::core::cap::request req_generic,
                 fractos::core::cap::request req_module_data,
                 fractos::core::cap::request req_ctx_sync,
                 fractos::core::cap::request req_ctx_destroy);

    std::string to_string(const Context& obj);
    std::string to_string(const ContextState& obj);

}
