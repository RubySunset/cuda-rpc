#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Stream : public impl::Base<srv::Stream, impl::Stream> {
        Stream(std::shared_ptr<fractos::core::channel> ch,
               fractos::wire::endian::uint32_t id,
               fractos::core::cap::request req_stream_sync,
               fractos::core::cap::request req_stream_destroy);

        std::weak_ptr<Stream> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::core::cap::request req_stream_sync;
        fractos::core::cap::request req_stream_destroy;
        fractos::wire::endian::uint32_t id;

        fractos::core::future<void> do_destroy();
    };

}
