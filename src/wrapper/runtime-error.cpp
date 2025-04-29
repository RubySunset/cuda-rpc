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

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaGetLastError()
{
    auto [err, state] = get_runtime_state_with_error();
    if (err) {
        return err;
    }

    err = state->last_error;
    state->last_error = cudaSuccess;
    return err;
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaPeekAtLastError()
{
    auto [err, state] = get_runtime_state_with_error();
    if (err) {
        return err;
    }

    return state->last_error;
}
