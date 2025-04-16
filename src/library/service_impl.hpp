#pragma once

#include <fractos/core/cap.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>


namespace impl {

    namespace srv = fractos::service::compute::cuda;

    struct Service {
        static Service& get(srv::Service& service);
        static const Service& get(const srv::Service& service);

        Service(std::shared_ptr<fractos::core::channel> ch,
                fractos::core::cap::request req_connect,
                fractos::core::cap::request req_generic);

        std::weak_ptr<Service> self;
        std::shared_ptr<fractos::core::channel> ch;

        // NOTE: keep connection as a separate request
        fractos::core::cap::request req_connect;
        fractos::core::cap::request req_generic;
    };

    std::string to_string(const Service& obj);

}
