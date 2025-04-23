#include <cuda.h>
#include <lib.hpp>

#include <../library/context_impl.hpp>


// * context management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    auto& state = get_state();
    auto device = state.get_device(dev);

    auto ctx = device->make_context(flags).get();
    {
        auto contexts_lock = std::unique_lock(state.contexts_mutex);
        auto res = state.contexts.insert(std::make_pair(ctx->get_context(), ctx));
        CHECK(res.second);
    }
    *pctx = ctx->get_context();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxGetCurrent(CUcontext* pctx)
{
    auto& state = get_state();
    auto& stack = state.get_context_stack();
    if (stack.empty()) {
        *pctx = nullptr;
    } else {
        auto ctx = stack.top();
        *pctx = ctx->get_context();
    }

    return CUDA_SUCCESS;
}

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

extern "C" [[gnu::visibility("default")]]
CUresult
cuCtxPopCurrent_v2(CUcontext *pctx)
{
    auto& state = get_state();
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
    auto& state = get_state();

    if (not ctx) {
        return CUDA_ERROR_INVALID_VALUE;
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
    auto& state [[maybe_unused]] = get_state();

    CUcontext cur;
    auto err = cuCtxPopCurrent(&cur);
    CHECK(err == CUDA_SUCCESS or err == CUDA_ERROR_INVALID_CONTEXT);

    err = cuCtxPushCurrent(ctx);
    CHECK(err == CUDA_SUCCESS or err == CUDA_ERROR_INVALID_VALUE);

    return CUDA_SUCCESS;
}
