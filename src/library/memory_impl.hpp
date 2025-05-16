#pragma once

#include <cuda.h>
#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct MemoryState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Memory> self;
        fractos::core::cap::request req_mem_destroy;
        CUdeviceptr addr;
        size_t size;
        fractos::core::cap::memory memory;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Memory = fractos::common::service::ImplWrapper<clt::Memory, impl::MemoryState>;

    std::shared_ptr<clt::Memory>
    make_memory(std::shared_ptr<fractos::core::channel> ch,
                CUdeviceptr addr,
                size_t size,
                fractos::core::cap::request req_mem_destroy,
                fractos::core::cap::memory memory);

    std::string to_string(const Memory& obj);
    std::string to_string(const MemoryState& obj);

}
