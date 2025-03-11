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
        
        

        struct Memory_impl {
            static Memory_impl& get(fractos::service::compute::cuda::Memory& memory);
            static const Memory_impl& get(const fractos::service::compute::cuda::Memory& memory);
        
            std::weak_ptr<Memory_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            
            fractos::core::cap::request req_mem_destroy;
        
            bool destroyed;
            char* addr;
            size_t size;
            fractos::core::cap::memory memory;

        };
        

        

    } // namespace cuda

