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
        
       
        struct cuda_device_impl {
            static cuda_device_impl& get(fractos::service::compute::cuda::cuda_device& device);
            static const cuda_device_impl& get(const fractos::service::compute::cuda::cuda_device& device);
        
            std::weak_ptr<cuda_device_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            fractos::core::cap::request req_make_cuda_context; // new
            // fractos::core::cap::request req_test;
            fractos::core::cap::request req_destroy;
        
            bool destroyed;
        };

        
        

    } // namespace cuda

