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
        
       
        struct Device_impl {
            static Device_impl& get(fractos::service::compute::cuda::Device& device);
            static const Device_impl& get(const fractos::service::compute::cuda::Device& device);
        
            std::weak_ptr<Device_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            fractos::core::cap::request req_make_context; // new
            // fractos::core::cap::request req_test;
            fractos::core::cap::request req_destroy;
        
            bool destroyed;
        };

        
        

    } // namespace cuda

