#pragma once

#include <chrono>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>


using namespace fractos;


namespace impl {
    class Service;
    class Context;
}

namespace impl {

    class Device : public fractos::common::service::SrvBase {
    public:
        int get_remote_cuordinal() const;
        CUdevice get_remote_cudevice() const;

        CUdevice cudevice;
        std::shared_ptr<Service> service;
        std::weak_ptr<Device> self;

        // NOTE: for internal use
    public:
        int _remote_cuordinal;
        fractos::core::cap::request _req_generic;

        void handle_generic(auto ch, auto args);
    protected:
        void handle_get_attribute(auto ch, auto args);
        void handle_get_name(auto ch, auto args);
        void handle_get_uuid(auto ch, auto args);
        void handle_total_mem(auto ch, auto args);
        void handle_get_properties(auto ch, auto args);
        void handle_ctx_create(auto ch, auto args);
        void handle_destroy(auto ch, auto args);
    };

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Device>>>
    make_device(std::shared_ptr<fractos::core::channel> ch,
                std::shared_ptr<Service> service,
                int cuordinal);

    std::string to_string(const Device& obj);

}
