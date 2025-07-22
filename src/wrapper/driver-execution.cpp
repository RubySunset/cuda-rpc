#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>

namespace srv = fractos::service::compute::cuda;


// * execution control
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html


extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuLaunchKernel(CUfunction f,
               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream,
               void **kernelParams, void **extra)
{
    auto& state = get_driver_state();

    auto func_ptr = state.get_function(f);
    if (not func_ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::shared_ptr<srv::Stream> stream_ptr; // NOTE: keep the object around
    std::optional<std::reference_wrapper<srv::Stream>> stream_opt;
    if (hStream) {
        stream_ptr = state.get_stream(hStream);
        if (not stream_ptr) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        stream_opt = *stream_ptr;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        func_ptr->function->launch((const void**)kernelParams,
                                   dim3(gridDimX, gridDimY, gridDimZ),
                                   dim3(blockDimX, blockDimY, blockDimZ),
                                   sharedMemBytes, stream_opt)
            .get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}
