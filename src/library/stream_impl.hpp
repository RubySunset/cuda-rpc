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
        
        struct Stream_impl {
            static Stream_impl& get(fractos::service::compute::cuda::Stream& stream);
            static const Stream_impl& get(const fractos::service::compute::cuda::Stream& stream);
        
            std::weak_ptr<Stream_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
            
            fractos::wire::endian::uint8_t error;

            fractos::core::cap::request req_stream_sync;
            fractos::core::cap::request req_stream_destroy;
            fractos::wire::endian::uint32_t id;
        
            bool destroyed;
            

        };

    } // namespace cuda

