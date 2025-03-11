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
        
        struct Service_impl {
            static Service_impl& get(fractos::service::compute::cuda::Service& service);
            static const Service_impl& get(const fractos::service::compute::cuda::Service& service);
        
            std::weak_ptr<Service_impl> self;
            std::shared_ptr<fractos::core::channel>
         ch;
            fractos::core::cap::request req_make_device;
        };

        

        

    } // namespace cuda

