#include <cuda.h>

#include <./driver-syms-extern.hpp>
#include <./state.hpp>


// * initialization
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html#group__CUDA__INITIALIZE

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuInit(unsigned int flags)
{
    CHECK((*ptr_cuInit)(flags) == CUDA_SUCCESS);

    auto& state = get_state();
    state.service->init(flags).get();
    return CUDA_SUCCESS;
}
