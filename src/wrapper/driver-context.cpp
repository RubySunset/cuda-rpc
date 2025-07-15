#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>


// * context management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    auto& state = get_driver_state();
    auto device = state.get_device(dev);

    auto ctx = device->make_context(flags).get();
    state.insert_context(ctx);
    *pctx = ctx->get_context();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
    auto& state = get_driver_state();
    auto ctx_ptr = state.get_context(ctx);
    if (not ctx_ptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *version = ctx_ptr->get_api_version().get();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxGetCurrent(CUcontext* pctx)
{
    auto& state = get_driver_state();
    auto ctx = state.get_current_context();
    if (ctx) {
        *pctx = ctx->get_context();
    } else {
        *pctx = nullptr;
    }

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxGetDevice(CUdevice *device)
{
    auto& state = get_driver_state();

    CUcontext ctx_id;
    CHECK(cuCtxGetCurrent(&ctx_id) == CUDA_SUCCESS);

    auto ctx = state.get_context(ctx_id);
    auto dev = ctx->get_device();
    *device = dev->get_device();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxGetLimit(size_t *pvalue, CUlimit limit)
{
    auto& state = get_driver_state();

    auto ctx_ptr = state.get_current_context();
    CHECK(ctx_ptr);

    *pvalue = ctx_ptr->get_limit(limit).get();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxPopCurrent_v2(CUcontext *pctx)
{
    auto& state = get_driver_state();
    auto& stack = state.get_context_stack();
    if (not stack.empty()) [[likely]] {
        *pctx = stack.top()->get_context();
        return CUDA_SUCCESS;
    } else {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxPushCurrent_v2(CUcontext ctx)
{
    auto& state = get_driver_state();

    if (not ctx) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    auto ctx_ptr = state.get_context(ctx);
    CHECK(ctx_ptr);

    auto& stack = state.get_context_stack();
    stack.push(ctx_ptr);

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxSetCurrent(CUcontext ctx)
{
    auto& state [[maybe_unused]] = get_driver_state();

    CUcontext cur;
    auto err = cuCtxPopCurrent(&cur);
    CHECK(err == CUDA_SUCCESS or err == CUDA_ERROR_INVALID_CONTEXT);

    err = cuCtxPushCurrent(ctx);
    CHECK(err == CUDA_SUCCESS or err == CUDA_ERROR_INVALID_CONTEXT);

    return err;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxSynchronize()
{
    auto& state = get_driver_state();

    auto ctx_ptr = state.get_current_context();
    ctx_ptr->synchronize().get();

    return CUDA_SUCCESS;
}
