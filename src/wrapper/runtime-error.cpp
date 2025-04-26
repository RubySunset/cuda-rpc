#include <cuda.h>

#include <./runtime-state.hpp>
#include <./runtime-syms-extern.hpp>


// * error handling
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html

extern "C" [[gnu::visibility("default")]]
const char* CUDARTAPI
cudaGetErrorName(cudaError_t error)
{
    return (*ptr_cudaGetErrorName)(error);
}

extern "C" [[gnu::visibility("default")]]
const char* CUDARTAPI
cudaGetErrorString(cudaError_t error)
{
    return (*ptr_cudaGetErrorString)(error);
}
