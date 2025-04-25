#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>


// * primary context management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html

extern "C" [[gnu::visibility("default")]]
CUresult
cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev)
{
    auto& state = get_driver_state();
    auto ctx = state.get_device_primary_context(dev);
    if (not ctx) {
        return CUDA_ERROR_INVALID_DEVICE;
    }
    *pctx = ctx->get_context();
    return CUDA_SUCCESS;
}
