#include <cuda.h>
#include <lib.hpp>

#include <../library/context_impl.hpp>


// * context management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxGetDevice(CUdevice *device)
{
    auto& state = get_state();

    CUcontext ctx_id;
    CHECK(cuCtxGetCurrent(&ctx_id) == CUDA_SUCCESS);

    auto ctx = state.get_context(ctx_id);
    auto dev = ctx->get_device();
    *device = dev->get_device();
    return CUDA_SUCCESS;
}
