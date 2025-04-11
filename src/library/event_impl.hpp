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
        
        struct Event_impl {
            static Event_impl& get(fractos::service::compute::cuda::Event& event);
            static const Event_impl& get(const fractos::service::compute::cuda::Event& event);
        
            std::weak_ptr<Event_impl> self;
            std::shared_ptr<fractos::core::channel> ch;
            
            fractos::wire::endian::uint8_t error;

            // fractos::core::cap::request req_event_sync;
            fractos::core::cap::request req_event_destroy;
        
            bool destroyed;
            

        };

    } // namespace cuda

