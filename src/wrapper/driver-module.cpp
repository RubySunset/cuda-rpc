#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>


// * module management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html

extern "C" [[gnu::visibility("default")]]
CUresult
cuModuleGetLoadingMode(CUmoduleLoadingMode* mode)
{
    auto& state = get_driver_state();
    *mode = state.service->module_get_loading_mode().get();
    return CUDA_SUCCESS;
}
