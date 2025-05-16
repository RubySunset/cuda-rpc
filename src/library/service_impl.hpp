#pragma once

#include <fractos/common/service/impl_base.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/future.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace clt = fractos::service::compute::cuda;

    struct ServiceState : public fractos::common::service::ImplState {
        std::weak_ptr<clt::Service> self;
        // NOTE: keep connection as a separate request
        fractos::core::cap::request req_connect;
        fractos::core::cap::request req_generic;

        fractos::core::future<void>
        do_destroy(std::shared_ptr<fractos::core::channel>& ch);
    };

    using Service = fractos::common::service::ImplWrapper<clt::Service, impl::ServiceState>;

    std::shared_ptr<clt::Service>
    make_service(std::shared_ptr<fractos::core::channel> ch,
                 fractos::core::cap::request req_connect,
                 fractos::core::cap::request req_generic);

    std::string to_string(const Service& obj);
    std::string to_string(const ServiceState& obj);

}
