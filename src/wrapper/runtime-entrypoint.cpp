#include "runtime-state.hpp"
#include <cuda_runtime.h>

#include <./runtime-state.hpp>
#include <./runtime-syms-extern.hpp>

// * driver entry point access
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DRIVER__ENTRY__POINT.html

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaGetDriverEntryPoint(
    const char *symbol,
    void **funcPtr,
    unsigned long long flags,
    cudaDriverEntryPointQueryResult* driverStatus)
{
    auto& state = get_runtime_state();

    int version;
    auto err = cudaDriverGetVersion(&version);
    return_error_maybe(err);

    err = cudaGetDriverEntryPointByVersion(symbol, funcPtr, version, flags, driverStatus);
    return_error(err);
}
