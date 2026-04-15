#include <atomic>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <glog/logging.h>

#include <./runtime-state.hpp>
#include <mutex>
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
    auto& global = state.global;

    std::unique_lock lock(global->kernel_mutex);
    auto it = global->addr_to_kernel.find(addr);
    if (it != global->addr_to_kernel.end()) {
        *kernel = it->second;
        return cudaSuccess;
    }
    uintptr_t curr_kernel_cnt = global->kernel_cnt++;
    *kernel = (cudaKernel_t)curr_kernel_cnt;
    global->kernel_to_addr[*kernel] = addr;
    global->addr_to_kernel[addr] = *kernel;
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
    auto& global = state.global;

    std::unique_lock lock(global->kernel_mutex);
    auto it = global->kernel_to_addr.find(kernel);
    if (it == global->kernel_to_addr.end()) {
        LOG(WARNING) << "Unable to find stored device function address for given cudaKernel";
        return cudaErrorInvalidDeviceFunction;
    }
    return cudaLaunchKernel(it->second, gridDim, blockDim, args, sharedMem, stream);
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
