#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

#include "common.hpp"


namespace impl {

    namespace clt = ::fractos::service::compute::cuda;

    struct EventState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Event> self;
        CUevent cuevent;

        fractos::core::cap::request req_generic;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Event = fractos::common::service::ImplWrapper<clt::Event, impl::EventState>;

    std::shared_ptr<clt::Event>
    make_event(std::shared_ptr<fractos::core::channel> ch,
               CUevent cuevent,
               fractos::core::cap::request req_generic);

    std::string to_string(const Event& obj);
    std::string to_string(const EventState& obj);
}
