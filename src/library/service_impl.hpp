#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/future.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>

#include <common.hpp>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct ServiceState {
        // NOTE: keep connection as a separate request
        fractos::core::cap::request req_connect;
        fractos::core::cap::request req_generic;
    };

    struct Service : public impl::Base<srv::Service, impl::Service> {
        std::weak_ptr<Service> self;
        std::shared_ptr<fractos::core::channel> ch;
        std::shared_ptr<ServiceState> state;

        fractos::core::future<void> do_destroy();
    };

    std::shared_ptr<Service> make_service(
        std::shared_ptr<fractos::core::channel> ch,
        fractos::core::cap::request req_connect,
        fractos::core::cap::request req_generic);

    std::shared_ptr<Service> make_service(
        std::shared_ptr<fractos::core::channel> ch,
        std::shared_ptr<ServiceState> state);

    std::string to_string(const Service& obj);

}
