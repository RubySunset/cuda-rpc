#pragma once

#include <fractos/logging.hpp>


#define LOG_OP(func)                            \
    VLOG(fractos::logging::SERVICE) << func
#define LOG_METHOD(method)                      \
    VLOG(fractos::logging::SERVICE) << to_string(*this) << "::" << method
#define LOG_REQ(method)                         \
    LOG_METHOD(method) << " <-"
#define LOG_RES(method)                         \
    LOG_METHOD(method) << " ->"
#define LOG_METHOD_PTR(method, ptr)                                     \
    VLOG(fractos::logging::SERVICE) << to_string(*ptr) << "::" << method
#define LOG_REQ_PTR(method, ptr)                \
    LOG_METHOD_PTR(method, ptr) << " <-"
#define LOG_RES_PTR(method, ptr)                \
    LOG_METHOD_PTR(method, ptr) << " ->"

#define METHOD(cls, name)                                               \
    static const std::string method = #name;                            \
    namespace srv_wire = ::service::compute::cuda::wire:: cls;          \
    using msg = srv_wire:: name;

#define CHECK_ARGS_EXACT()                                              \
    if (not args->has_exactly_args()) {                                 \
        throw core::other_error("invalid response format for " + method); \
    }                                                                   \
    CHECK_ARGS_ERROR()

#define CHECK_ARGS_CAPS_EXACT()                                         \
    if (not args->has_exactly_caps()) {                                 \
        throw core::other_error("invalid response format for " + method); \
    }

#define CHECK_ARGS_IMMS_MIN()                                           \
    if (not args->has_all_imms()) {                                     \
        throw core::other_error("invalid response format for " + method); \
    }

#define CHECK_ARGS_ERROR()                                              \
    fractos::wire::error_raise_exception_maybe(args->imms.error);


#define then_cuda_response()                                            \
    then([this, self](auto& fut) {                                      \
        auto [ch, args] = fut.get();                                    \
        LOG_RES_PTR(method, self)                                       \
            << wire::to_string(*args);                                  \
        if (not args->has_imm(&msg::response::imms::error)) [[unlikely]] { \
            throw core::other_error("invalid response format for " + method); \
        }                                                               \
        fractos::wire::error_raise_exception_maybe(args->imms.error);   \
        if (not args->has_imm(&msg::response::imms::cuerror)) [[unlikely]] { \
            throw core::other_error("invalid response format for " + method); \
        }                                                               \
        if (args->imms.cuerror) {                                       \
            throw fractos::service::compute::cuda::CudaError((CUresult)args->imms.cuerror.get()); \
        }                                                               \
        return std::make_pair(std::move(ch), std::move(args));          \
    })


namespace impl {

    template<class Tsrv, class Timpl>
    class Base {
    public:
        static Timpl& get(Tsrv& obj);
        static const Timpl& get(const Tsrv& obj);

        fractos::core::future<void> destroy();
        fractos::core::future<bool> destroy_maybe();
        virtual fractos::core::future<void> do_destroy() = 0;

    private:
        std::atomic_flag _destroyed;
    };

}

#include <./common.inc.hpp>
