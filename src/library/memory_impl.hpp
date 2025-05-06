#pragma once

#include <cuda.h>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Memory : public impl::Base<srv::Memory, impl::Memory> {
        Memory(std::shared_ptr<fractos::core::channel> ch,
               CUdeviceptr addr, size_t size,
               fractos::core::cap::request req_mem_destroy,
               fractos::core::cap::memory memory);

        std::weak_ptr<Memory> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::core::cap::request req_mem_destroy;

        CUdeviceptr addr;
        size_t size;
        fractos::core::cap::memory memory;

        fractos::core::future<void> do_destroy();
    };

}
