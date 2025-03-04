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
        
        struct cuda_service_impl {
            static cuda_service_impl& get(fractos::service::compute::cuda::cuda_service& service);
            static const cuda_service_impl& get(const fractos::service::compute::cuda::cuda_service& service);
        
            std::weak_ptr<cuda_service_impl> self;
            std::shared_ptr<fractos::core::channel>
         ch;
            fractos::core::cap::request req_make_cuda_device;
        };

        

        

    } // namespace cuda

