#include <cuda.h>

#include <./driver-state.hpp>

namespace srv = fractos::service::compute::cuda;


// * execution control
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html


extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib, int  value)
{
    auto& state = get_driver_state();

    auto func_ptr = state.get_function(hfunc);
    if (not func_ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        func_ptr->function->set_attribute(attrib,  value)
            .get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuLaunchKernel(CUfunction f,
               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream,
               void **kernelParams, void **extra)
{
    if (extra != NULL) {
        LOG_FIRST_N(ERROR, 1) << "TODO: add support for kernel launch args specified via the extra parameter";
    }

    auto& state = get_driver_state();

    auto func_ptr = state.get_function(f);
    if (not func_ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::shared_ptr<srv::Stream> stream_ptr = state.get_stream(hStream);
    if (not stream_ptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    CUresult error = CUDA_SUCCESS;
    try {
        func_ptr->function->launch((const void**)kernelParams,
                                   dim3(gridDimX, gridDimY, gridDimZ),
                                   dim3(blockDimX, blockDimY, blockDimZ),
                                   sharedMemBytes, *stream_ptr)
            .get();
    } catch (const srv::CudaError& e) {
        error = e.cuerror;
    }

    return error;
}
