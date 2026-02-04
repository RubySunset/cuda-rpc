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
        CUdeviceptr cudeviceptr;
        size_t size;
        fractos::core::cap::memory memory;

        fractos::core::cap::request req_generic;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Memory = fractos::common::service::ImplWrapper<clt::Memory, impl::MemoryState>;

    std::shared_ptr<clt::Memory>
    make_memory(std::shared_ptr<fractos::core::channel> ch,
                CUdeviceptr cudeviceptr,
                size_t size,
                fractos::core::cap::memory memory,
                fractos::core::cap::request req_generic);

    std::string to_string(const Memory& obj);
    std::string to_string(const MemoryState& obj);

}
