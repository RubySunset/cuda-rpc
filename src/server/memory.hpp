#include <chrono>
#include <cuda.h>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>


namespace impl {
    class Context;
}

namespace impl {

    class Memory : public fractos::common::service::SrvBase {
    public:
        CUdeviceptr cuptr;
        fractos::core::cap::memory memory;
        std::unique_ptr<fractos::core::memory_region> mr;
        std::weak_ptr<Context> ctx;
        std::shared_ptr<Memory> self; // NOTE: keep object alive

        // NOTE: for internal use
    public:
        fractos::core::cap::request req_generic;

        void handle_generic(auto ch, auto args);
    protected:
        void handle_destroy(auto ch, auto args);
    };

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Memory>>>
    make_memory(std::shared_ptr<fractos::core::channel> ch, std::shared_ptr<Context> ctx, size_t size);

    std::string to_string(const Memory& obj);
}
