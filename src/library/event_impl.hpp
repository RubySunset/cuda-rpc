#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Event {
        static Event& get(srv::Event& event);
        static const Event& get(const srv::Event& event);

        std::weak_ptr<Event> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::wire::endian::uint8_t error;

        // fractos::core::cap::request req_event_sync;
        fractos::core::cap::request req_event_destroy;

        bool destroyed;
    };

}
