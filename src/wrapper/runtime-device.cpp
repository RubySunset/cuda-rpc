#include "runtime-state.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

#include <./driver-state.hpp>
#include <./runtime-syms-extern.hpp>


// * device management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    auto& state [[maybe_unused]] = get_runtime_state();

    CUdevice dev;
    auto err = (cudaError_t)cuDeviceGet(&dev, device);
    return_error_maybe(err);

    err = (cudaError_t)cuDeviceGetAttribute(value, (CUdevice_attribute)attr, device);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaDeviceSynchronize()
{
    auto& state = get_runtime_state();

    auto err = (cudaError_t)cuCtxSynchronize();
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaGetDevice(int* device)
{
    auto& state [[maybe_unused]] = get_runtime_state();
    *device = state.dev_o;
    return_error(cudaSuccess);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaGetDeviceCount(int *count)
{
    auto& state [[maybe_unused]] = get_runtime_state();
    auto err = (cudaError_t)cuDeviceGetCount(count);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaSetDevice(int device)
{
    auto& state = get_runtime_state();

    CUdevice dev;
    auto err = (cudaError_t)cuDeviceGet(&dev, device);
    return_error_maybe(err);

    state.dev_o = device;
    state.dev = dev;

    CUcontext ctx;
    err = (cudaError_t)cuDevicePrimaryCtxRetain(&ctx, dev);
    return_error_maybe(err);

    err = (cudaError_t)cuCtxSetCurrent(ctx);
    return_error(cudaSuccess);
}
