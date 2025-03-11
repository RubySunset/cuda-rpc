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
        
        

        struct Module_impl {
            static Module_impl& get(fractos::service::compute::cuda::Module& module);
            static const Module_impl& get(const fractos::service::compute::cuda::Module& module);
        
            std::weak_ptr<Module_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;
            fractos::core::cap::request req_get_func; 
            fractos::core::cap::request req_module_unload;
        
            bool destroyed;
        };
        

        

    } // namespace cuda

