#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

#include "./common.hpp"


namespace impl {

    namespace clt = ::fractos::service::compute::cuda;

    struct LibraryState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Library> self;
        CUlibrary culibrary;

        fractos::core::cap::request req_generic;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Library = fractos::common::service::ImplWrapper<clt::Library, impl::LibraryState>;

    std::shared_ptr<clt::Library>
    make_library(std::shared_ptr<fractos::core::channel> ch,
                 CUlibrary culibrary,
                 fractos::core::cap::request req_generic);

    std::string to_string(const Library& obj);
    std::string to_string(const LibraryState& obj);
}
