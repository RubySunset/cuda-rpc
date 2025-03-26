#pragma once

#include <any>
#include <cuda.h>
#include <fractos/core/future.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/gns.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace fractos::service::compute::cuda;

namespace impl { 
        

        struct Context_impl {
            static Context_impl& get(fractos::service::compute::cuda::Context& context);
            static const Context_impl& get(const fractos::service::compute::cuda::Context& context);
        
            std::weak_ptr<Context_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            fractos::core::cap::request req_memory;
            fractos::core::cap::request req_memory_rpc_test;
            fractos::core::cap::request req_stream;
            fractos::core::cap::request req_module_data; // file
            fractos::core::cap::request req_ctx_sync;
            fractos::core::cap::request req_ctx_destroy;
        
            bool destroyed;
        };


        

    } // namespace cuda

