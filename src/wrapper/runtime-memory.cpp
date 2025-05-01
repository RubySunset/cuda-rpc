#include "runtime-state.hpp"
#include <cuda_runtime.h>

#include "./runtime-state.hpp"
#include "./runtime-syms-extern.hpp"


// * memory management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaFree(void* devPtr)
{
    auto& state = get_runtime_state();

    if (devPtr == nullptr) {
        return cudaSuccess;
    }

    auto err = (cudaError_t)cuMemFree((CUdeviceptr)devPtr);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMalloc(void** devPtr, size_t size)
{
    auto& state = get_runtime_state();
    auto err = (cudaError_t)cuMemAlloc((CUdeviceptr*)devPtr, size);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    switch (kind) {
    case cudaMemcpyHostToHost:
        return (cudaError_t)cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count);
        break;
    case cudaMemcpyHostToDevice:
        return (cudaError_t)cuMemcpyHtoD((CUdeviceptr)dst, src, count);
        break;
    case cudaMemcpyDeviceToHost:
        return (cudaError_t)cuMemcpyDtoH(dst, (CUdeviceptr)src, count);
        break;
    case cudaMemcpyDeviceToDevice:
        return (cudaError_t)cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count);
        break;
    case cudaMemcpyDefault:
        return (cudaError_t)cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count);
        break;
    default:
        return cudaErrorInvalidValue;
    }
}
