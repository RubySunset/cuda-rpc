#pragma once

#include <fractos/common/service/clt_impl.hpp>
#include <fractos/service/compute/cuda_msg.hpp>


// NOTE: only capture this, assuming you'll capture self in a later continuation

#define then_check_response() then_check_response_ptr(this)

#define then_check_response_ptr(_ptr_)                                  \
    then_log_response_ptr(_ptr_)                                        \
    .then([ptr=_ptr_](auto& fut) {                                      \
        auto [ch, args] = fut.get();                                    \
        CHECK_ARGS_COND(args->has_imm(&std::decay_t<decltype(*args)>::base_type::imms::error)); \
        fractos::wire::error_raise_exception_maybe(args->imms.error);   \
        return std::make_pair(std::move(ch), std::move(args));          \
    })

#define then_check_cuda_response() then_check_cuda_response_ptr(this)

#define then_check_cuda_response_ptr(_ptr_)                             \
    then_check_response_ptr(_ptr_)                                      \
    .then([ptr=_ptr_](auto& fut) {                                      \
        auto [ch, args] = fut.get();                                    \
        CHECK_ARGS_COND(args->has_imm(&std::decay_t<decltype(*args)>::base_type::imms::cuerror)); \
        if (args->imms.cuerror) {                                       \
            throw fractos::service::compute::cuda::CudaError((CUresult)args->imms.cuerror.get()); \
        }                                                               \
        return std::make_pair(std::move(ch), std::move(args));          \
    })

#define then_check_cublas_response() then_check_cublas_response_ptr(this)

#define then_check_cublas_response_ptr(_ptr_)                             \
    then_check_response_ptr(_ptr_)                                      \
    .then([ptr=_ptr_](auto& fut) {                                      \
        auto [ch, args] = fut.get();                                    \
        CHECK_ARGS_COND(args->has_imm(&std::decay_t<decltype(*args)>::base_type::imms::cublas_error)); \
        if (args->imms.cublas_error) {                                       \
            throw fractos::service::compute::cuda::CublasError((cublasStatus_t)args->imms.cublas_error.get()); \
        }                                                               \
        return std::make_pair(std::move(ch), std::move(args));          \
    })
