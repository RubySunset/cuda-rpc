#pragma once

#include <any>
#include <cuda.h>
#include <fractos/core/future.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Context;
    struct Context : public impl::Base<srv::Context, impl::Context> {
        Context(std::shared_ptr<fractos::core::channel> ch,
                std::shared_ptr<srv::Device> device,
                fractos::core::cap::request req_generic,
                fractos::core::cap::request req_memory,
                fractos::core::cap::request req_memory_rpc_test,
                fractos::core::cap::request req_stream,
                fractos::core::cap::request req_event,
                fractos::core::cap::request req_module_data,
                fractos::core::cap::request req_ctx_sync,
                fractos::core::cap::request req_ctx_destroy);

        std::weak_ptr<srv::Context> self;
        std::shared_ptr<fractos::core::channel> ch;

        CUcontext context;
        std::weak_ptr<srv::Device> device;

        fractos::core::cap::request req_generic;
        fractos::core::cap::request req_memory;
        fractos::core::cap::request req_memory_rpc_test;
        fractos::core::cap::request req_stream;
        fractos::core::cap::request req_event;
        fractos::core::cap::request req_module_data; // file
        fractos::core::cap::request req_ctx_sync;
        fractos::core::cap::request req_ctx_destroy;

        bool destroyed;

        // an opaque data structure in libcuda
        std::unique_ptr<char[]> context_ptr;

        fractos::core::future<void> do_destroy();
    };

    std::string to_string(const Context& obj);

}
