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
        
        

        struct cuda_function_impl {
            static cuda_function_impl& get(fractos::service::compute::cuda::cuda_function& func);
            static const cuda_function_impl& get(const fractos::service::compute::cuda::cuda_function& func);
        
            std::weak_ptr<cuda_function_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            // fractos::core::cap::request req_test; refer to initialize
            fractos::core::cap::request req_func_call;
            fractos::core::cap::request req_func_destroy;
        
            bool destroyed;
        };
        

        

    } // namespace cuda

