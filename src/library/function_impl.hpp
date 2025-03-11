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
        
        

        struct Function_impl {
            static Function_impl& get(fractos::service::compute::cuda::Function& func);
            static const Function_impl& get(const fractos::service::compute::cuda::Function& func);
        
            std::weak_ptr<Function_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
        
            fractos::wire::endian::uint8_t error;

            fractos::core::cap::request req_func_call;
            fractos::core::cap::request req_func_destroy;
        
            bool destroyed;
        };
        

        

    } // namespace cuda

