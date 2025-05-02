#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>

namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Event : public impl::Base<srv::Event, impl::Event> {
        Event(std::shared_ptr<fractos::core::channel> ch,
              fractos::wire::endian::uint8_t error,
              fractos::core::cap::request req_event_destroy);

        std::weak_ptr<Event> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::wire::endian::uint8_t error;

        // fractos::core::cap::request req_event_sync;
        fractos::core::cap::request req_event_destroy;

        fractos::core::future<void> do_destroy();
    };

}
