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
        

        struct cuda_context_impl {
            static cuda_context_impl& get(fractos::service::compute::cuda::cuda_context& context);
            static const cuda_context_impl& get(const fractos::service::compute::cuda::cuda_context& context);
        
            std::weak_ptr<cuda_context_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            fractos::core::cap::request req_cuMemalloc;
            fractos::core::cap::request req_ctx_sync;
            fractos::core::cap::request req_ctx_destroy;
        
            bool destroyed;
        };


        

    } // namespace cuda

