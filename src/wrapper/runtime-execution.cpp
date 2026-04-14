#include <atomic>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include <./runtime-state.hpp>
#include <runtime-lib.hpp>


// * execution control
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem,
                            struct CUstream_st *stream)
{
    return (*get_runtime_lib_syms().ptr___cudaPushCallConfiguration)(gridDim, blockDim, sharedMem, stream);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem,
                           void *stream)
{
    return (*get_runtime_lib_syms().ptr___cudaPopCallConfiguration)(gridDim, blockDim, sharedMem, stream);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaGetKernel(cudaKernel_t* kernel, const void* addr)
{
    auto& state = get_runtime_state();

    uintptr_t curr_kernel_cnt = state.global->kernel_cnt.fetch_add(1, std::memory_order::relaxed);
    *kernel = (cudaKernel_t)curr_kernel_cnt;
    state.global->kernel_trans[curr_kernel_cnt] = addr;
    return cudaSuccess;
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
__cudaLaunchKernel(
        cudaKernel_t kernel,
        dim3 gridDim,
        dim3 blockDim,
        void **args,
        size_t sharedMem,
        cudaStream_t stream)
{
    auto& state = get_runtime_state();
    const void* addr = state.global->kernel_trans[(uintptr_t)kernel];
    return cudaLaunchKernel(addr, gridDim, blockDim, args, sharedMem, stream);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaFuncSetAttribute(const void* func, cudaFuncAttribute attr, int  value)
{
    auto& state = get_runtime_state();

    auto [err, cufunc] = state.get_function(func);
    return_error_maybe(err);

    err = (cudaError_t)cuFuncSetAttribute(cufunc, (CUfunction_attribute)attr, value);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                 size_t sharedMem, cudaStream_t stream)
{
    auto& state = get_runtime_state();

    auto [err, cufunc] = state.get_function(func);
    return_error_maybe(err);

    err = (cudaError_t)cuLaunchKernel(
        cufunc,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMem, stream, args, nullptr);
    return_error(err);
}
