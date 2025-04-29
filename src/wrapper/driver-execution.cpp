#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>

namespace srv = fractos::service::compute::cuda;


// * execution control
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html


extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuLaunchKernel(CUfunction f,
               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream,
               void **kernelParams, void **extra)
{
    auto& state = get_driver_state();

    auto func_ptr = state.get_function(f);
    if (not func_ptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::optional<std::reference_wrapper<srv::Stream>> stream;
    CHECK(not hStream);

    func_ptr->function->launch(dim3(gridDimX, gridDimY, gridDimZ),
                               dim3(blockDimX, blockDimY, blockDimZ),
                               (const void**)kernelParams, sharedMemBytes, stream)
        .get();

    return CUDA_SUCCESS;
}
