#include <cuda_runtime.h>
#include <glog/logging.h>

#include <./runtime-state.hpp>
#include <./runtime-syms-extern.hpp>


// * execution control
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            struct CUstream_st *stream)
{
    return (*ptr___cudaPushCallConfiguration)(gridDim, blockDim, sharedMem, stream);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                           void *stream)
{
    return (*ptr___cudaPopCallConfiguration)(gridDim, blockDim, sharedMem, stream);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                 size_t sharedMem, cudaStream_t stream)
{
    auto& state = get_runtime_state();

    std::shared_ptr<RuntimeState::func_desc> func_desc;
    {
        auto funcs_lock = std::shared_lock(state.global->funcs_mutex);
        auto it = state.global->funcs.find((uintptr_t)func);
        if (it == state.global->funcs.end()) {
            return_error(cudaErrorInvalidDeviceFunction);
        }
        func_desc = it->second;
    }

    auto cufunc = func_desc->func.load(std::memory_order_acquire);

    if (cufunc == 0) [[unlikely]] {
        auto func_lock = std::unique_lock(func_desc->func_mutex);
        if (func_desc->func == 0) {
            CUfunction cufunc;
            auto cuerr = cuModuleGetFunction(&cufunc, func_desc->module, func_desc->name.c_str());
            return_error_maybe((cudaError_t)cuerr);
            // note: std::atomic<T*> seems zeroes source
            func_desc->func.store(cufunc);
        }
        cufunc = func_desc->func.load();
    }

    auto err = (cudaError_t)cuLaunchKernel(
        cufunc,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem, stream, args, nullptr);
    return_error(err);
}
