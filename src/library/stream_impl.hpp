#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Stream {
        static Stream& get(srv::Stream& stream);
        static const Stream& get(const srv::Stream& stream);
        Stream(std::shared_ptr<fractos::core::channel> ch,
               fractos::wire::endian::uint8_t error,
               fractos::wire::endian::uint32_t id,
               fractos::core::cap::request req_stream_sync,
               fractos::core::cap::request req_stream_destroy);

        std::weak_ptr<Stream> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::wire::endian::uint8_t error;

        fractos::core::cap::request req_stream_sync;
        fractos::core::cap::request req_stream_destroy;
        fractos::wire::endian::uint32_t id;

        bool destroyed;
    };

}
