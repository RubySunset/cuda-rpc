#include <cuda.h>

#include <./driver-state.hpp>

namespace srv = fractos::service::compute::cuda;


// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html

extern "C" [[gnu::visibility("default")]]
CUresult
cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    auto& state = get_driver_state();

    auto stream_ptr = state.get_stream(stream);

    if (not stream_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        stream_ptr->wait_value_32(addr, value, flags).get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr, cuuint32_t value, unsigned int flags)
{
    auto& state = get_driver_state();

    auto stream_ptr = state.get_stream(stream);

    if (not stream_ptr) [[unlikely]] {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        stream_ptr->write_value_32(addr, value, flags).get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}
