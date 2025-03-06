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
        
        

        struct cuda_memory_impl {
            static cuda_memory_impl& get(fractos::service::compute::cuda::cuda_memory& memory);
            static const cuda_memory_impl& get(const fractos::service::compute::cuda::cuda_memory& memory);
        
            std::weak_ptr<cuda_memory_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            // fractos::core::cap::request req_test; refer to initialize
            fractos::core::cap::request req_mem_destroy;
        
            bool destroyed;
            char* addr;
            size_t size;
            fractos::core::cap::memory memory;

        };
        

        

    } // namespace cuda

