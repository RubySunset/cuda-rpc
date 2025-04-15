#pragma once

#include <any>
#include <cuda.h>
#include <fractos/core/future.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Context {
        static Context& get(srv::Context& context);
        static const Context& get(const srv::Context& context);

        std::weak_ptr<Context> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::wire::endian::uint8_t error;
        fractos::core::cap::request req_memory;
        fractos::core::cap::request req_memory_rpc_test;
        fractos::core::cap::request req_stream;
        fractos::core::cap::request req_event;
        fractos::core::cap::request req_module_data; // file
        fractos::core::cap::request req_ctx_sync;
        fractos::core::cap::request req_ctx_destroy;

        bool destroyed;
    };

}
