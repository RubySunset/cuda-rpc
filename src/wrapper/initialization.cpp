#include <cuda.h>

#include <./lib.hpp>


// * initialization
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE

decltype(&cuInit) ptr_cuInit;

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuInit(unsigned int flags)
{
    CHECK((*ptr_cuInit)(flags) == CUDA_SUCCESS);

    auto& state = get_state();
    state.service->init(flags).get();
    return CUDA_SUCCESS;
}
