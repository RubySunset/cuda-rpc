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
        return_error(cudaSuccess);
    }

    auto err = (cudaError_t)cuMemFree((CUdeviceptr)devPtr);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaGetSymbolAddress(void **devPtr, const void *symbol)
{
    auto& state = get_runtime_state();

    auto [err, address] = state.get_variable(symbol);
    return_error_maybe(err);

    *devPtr = (void*)address;
    return_error(cudaSuccess);
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
cudaMemGetInfo(size_t* free, size_t* total)
{
    auto& state = get_runtime_state();

    auto err = (cudaError_t)cuMemGetInfo(free, total);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    auto& state = get_runtime_state();

    switch (kind) {
    case cudaMemcpyHostToHost:
        return_error((cudaError_t)cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count));
        break;
    case cudaMemcpyHostToDevice:
        return_error((cudaError_t)cuMemcpyHtoD((CUdeviceptr)dst, src, count));
        break;
    case cudaMemcpyDeviceToHost:
        return_error((cudaError_t)cuMemcpyDtoH(dst, (CUdeviceptr)src, count));
        break;
    case cudaMemcpyDeviceToDevice:
        return((cudaError_t)cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, count));
        break;
    case cudaMemcpyDefault:
        return_error((cudaError_t)cuMemcpy((CUdeviceptr)dst, (CUdeviceptr)src, count));
        break;
    default:
        return_error(cudaErrorInvalidValue);
    }
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    auto& state = get_runtime_state();

    switch (kind) {
    case cudaMemcpyHostToHost:
        return_error((cudaError_t)cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream));
        break;
    case cudaMemcpyHostToDevice:
        return_error((cudaError_t)cuMemcpyHtoDAsync((CUdeviceptr)dst, src, count, stream));
        break;
    case cudaMemcpyDeviceToHost:
        return_error((cudaError_t)cuMemcpyDtoHAsync(dst, (CUdeviceptr)src, count, stream));
        break;
    case cudaMemcpyDeviceToDevice:
        return((cudaError_t)cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream));
        break;
    case cudaMemcpyDefault:
        return_error((cudaError_t)cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, count, stream));
        break;
    default:
        return_error(cudaErrorInvalidValue);
    }
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMemset(void *devPtr, int value, size_t count)
{
    auto& state = get_runtime_state();
    return_error((cudaError_t)cuMemsetD8((CUdeviceptr)devPtr, value&0xff, count));
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    auto& state = get_runtime_state();
    return_error((cudaError_t)cuMemsetD8Async((CUdeviceptr)devPtr, value&0xff, count, stream));
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    auto& state = get_runtime_state();
    return_error((cudaError_t)cuMemsetD2D8((CUdeviceptr)devPtr, pitch, value&0xff, width, height));
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaMemset2DAsync(void *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream)
{
    auto& state = get_runtime_state();
    return_error((cudaError_t)cuMemsetD2D8Async((CUdeviceptr)devPtr, pitch, value&0xff, width, height, stream));
}
