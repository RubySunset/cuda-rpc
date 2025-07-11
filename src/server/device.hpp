#pragma once

#include <chrono>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>


using namespace fractos;


namespace impl {
    class Context;
}

namespace impl {

    class Device : public fractos::common::service::SrvBase {
    public:
        const CUdevice device;
        std::weak_ptr<Context> ctx_ptr;
        std::weak_ptr<Device> self;

    protected:
        void handle_generic(auto ch, auto args);
        void handle_get_attribute(auto ch, auto args);
        void handle_get_name(auto ch, auto args);
        void handle_get_uuid(auto ch, auto args);
        void handle_total_mem(auto ch, auto args);
        void handle_get_properties(auto ch, auto args);
        void handle_ctx_create(auto ch, auto args);
        void handle_destroy(auto ch, auto args);

    public:
        fractos::core::cap::request req_generic;

    public:
        // NOTE: for internal use
        Device(int ordinal);
        ~Device();
        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);
    };

    std::pair<CUresult, std::shared_ptr<Device>> make_device(int ordinal);

    std::string to_string(const Device& obj);

}
