#include <cuda_runtime.h>
#include <glog/logging.h>

#include <./runtime-state.hpp>

// * version management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART____VERSION.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaDriverGetVersion(int *driverVersion)
{
    auto& state [[maybe_unused]] = get_runtime_state();
    auto err = (cudaError_t)cuDriverGetVersion(driverVersion);
    return_error(err);
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaRuntimeGetVersion(int *runtimeVersion)
{
    LOG_FIRST_N(WARNING, 1) << "TODO: returning driver version as runtime version";
    return cudaDriverGetVersion(runtimeVersion);
}
