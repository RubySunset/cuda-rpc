#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>


// * error handling
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetErrorName(CUresult error, const char **pStr)
{
    return (*ptr_cuGetErrorName)(error, pStr);
}
