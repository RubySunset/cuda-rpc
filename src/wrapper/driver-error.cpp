#include <cuda.h>

#include <driver-lib.hpp>


// * error handling
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__ERROR.html

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetErrorName(CUresult error, const char **pStr)
{
    return (*get_driver_lib_syms().ptr_cuGetErrorName)(error, pStr);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetErrorString(CUresult error, const char **pStr)
{
    return (*get_driver_lib_syms().ptr_cuGetErrorString)(error, pStr);
}
