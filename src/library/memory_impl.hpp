#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Memory : public impl::Base<srv::Memory, impl::Memory> {
        Memory(std::shared_ptr<fractos::core::channel> ch,
               fractos::wire::endian::uint8_t error,
               char* addr, size_t size,
               fractos::core::cap::request req_mem_destroy,
               fractos::core::cap::memory memory);

        std::weak_ptr<Memory> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::wire::endian::uint8_t error;

        fractos::core::cap::request req_mem_destroy;

        char* addr;
        size_t size;
        fractos::core::cap::memory memory;

        fractos::core::future<void> do_destroy();
    };

}
