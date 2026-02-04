#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "fractos/service/compute/cuda.hpp"
#include <fractos/logging.hpp>

#include <./cublas-state.hpp>

namespace clt = fractos::service::compute::cuda;


// https://docs.nvidia.com/cuda/cublas/


extern "C" [[gnu::visibility("default")]]
cublasStatus_t
cublasCreate(cublasHandle_t* handle)
{
    auto& cublas_state = get_cublas_state();

    if (handle == NULL) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    auto& driver_state = get_driver_state_return_cublas();
    auto ctx_ptr = driver_state.get_current_context();
    if (ctx_ptr == nullptr) [[unlikely]] {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    try {
        auto cublas_ptr = ctx_ptr->cublas_create().get();
        cublas_state.add_handle(cublas_ptr->get_handle(), cublas_ptr);
        *handle = cublas_ptr->get_handle();
        return CUBLAS_STATUS_SUCCESS;
    } catch (const srv::CublasError& e) {
        return e.cublas_error;
    }
}

extern "C" [[gnu::visibility("default")]]
cublasStatus_t
cublasDestroy(cublasHandle_t handle)
{
    auto& cublas_state = get_cublas_state();

    auto cublas_ptr = cublas_state.get_handle(handle);
    if (cublas_ptr == nullptr) [[unlikely]] {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    try {
        cublas_ptr->destroy().get();
        cublas_state.erase_handle(handle);
        cublas_state.erase_stream(handle);
        return CUBLAS_STATUS_SUCCESS;
    } catch (const srv::CublasError& e) {
        return e.cublas_error;
    }
}

extern "C" [[gnu::visibility("default")]]
cublasStatus_t
cublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    auto& cublas_state = get_cublas_state();

    auto cublas_ptr = cublas_state.get_handle(handle);
    if (cublas_ptr == nullptr) [[unlikely]] {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }

    auto& driver_state = get_driver_state_return_cublas();
    auto stream_ptr = driver_state.get_stream(streamId);
    if (stream_ptr != nullptr) [[likely]] {
        cublas_state.update_stream(handle, stream_ptr);
    }
    // Regardless of whether the stream handle is valid or not,
    // the real CUBLAS library will return a success and segfault
    // on the next CUBLAS kernel launch.
    // We emulate this behaviour by throwing a CudaException
    // if a CUBLAS kernel is launched with an invalid stream
    return CUBLAS_STATUS_SUCCESS;
}
