#include <chrono>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>

#include <cuda.h>
#include <cublas_v2.h>

#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>


namespace impl {
    class Context;
}

namespace impl {

    class CublasHandle : public fractos::common::service::SrvBase {
    public:
        cublasHandle_t get_remote_handle() const;

        const cublasHandle_t cublas_handle;
        std::shared_ptr<Context> ctx_ptr;
        std::shared_ptr<CublasHandle> self;

        fractos::core::future<std::pair<fractos::wire::error_type, cublasStatus_t>>
        destroy_maybe(auto ch);

    protected:
        void handle_generic(auto ch, auto args);
        void handle_autogen_func(auto ch, auto args);
        void handle_destroy(auto ch, auto args);

        // NOTE: for internal use
    public:
        fractos::core::cap::request req_generic;

        CublasHandle(std::shared_ptr<Context> ctx, cublasHandle_t handle);
        ~CublasHandle();
        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);
    };

    std::pair<std::shared_ptr<CublasHandle>, cublasStatus_t> make_cublas_handle(std::shared_ptr<Context> ctx);

    std::string to_string(const CublasHandle& obj);

}
