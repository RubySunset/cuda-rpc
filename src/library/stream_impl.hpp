#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct StreamState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Stream> self;
        std::shared_ptr<fractos::core::channel> ch;

        fractos::core::cap::request req_generic;
        CUstream custream;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Stream = fractos::common::service::ImplWrapper<clt::Stream, impl::StreamState>;

    std::shared_ptr<clt::Stream>
    make_stream(std::shared_ptr<fractos::core::channel> ch,
                CUstream custream,
                fractos::core::cap::request req_generic);

    std::string to_string(const Stream& obj);
    std::string to_string(const StreamState& obj);

}
