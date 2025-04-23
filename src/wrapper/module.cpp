#include <cuda.h>

#include <./state.hpp>


// * module management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html

extern "C" [[gnu::visibility("default")]]
CUresult
cuModuleGetLoadingMode(CUmoduleLoadingMode* mode)
{
    auto& state = get_state();
    *mode = state.service->module_get_loading_mode().get();
    return CUDA_SUCCESS;
}
