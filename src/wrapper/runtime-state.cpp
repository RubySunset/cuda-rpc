#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "./runtime-state.hpp"


std::mutex _runtime_state_mutex;
std::atomic<std::shared_ptr<RuntimeState>> _runtime_state;
boost::thread_specific_ptr<std::shared_ptr<RuntimeThreadState>> _runtime_thread_state;


cudaError_t
do_runtime_init()
{
    CUresult err = CUDA_SUCCESS;
    auto tstate = std::make_shared<RuntimeThreadState>();

    auto state = _runtime_state.load();
    if (state) {
        goto done_state;
    }

    // initialize global state

    {
        auto runtime_state_lock = std::unique_lock(_runtime_state_mutex);

        state = _runtime_state.load();
        if (state) {
            goto done_state;
        }

        state = std::make_shared<RuntimeState>();

        err = cuInit(0);
        if (err != CUDA_SUCCESS) {
            goto err_state;
        }

        int dev_count;
        err = cuDeviceGetCount(&dev_count);
        if (err != CUDA_SUCCESS) {
            goto err_state;
        }

        for (auto dev_idx = 0; dev_idx < dev_count; dev_idx++) {
            CUdevice dev;
            err = cuDeviceGet(&dev, dev_idx);
            if (err != CUDA_SUCCESS) {
                goto err_state;
            }

            CUcontext ctx;
            err = cuDevicePrimaryCtxRetain(&ctx, dev_idx);
            if (err != CUDA_SUCCESS) {
                goto err_state;
            }
        }

        _runtime_state = state;
    }

err_state:

done_state:

    // initialize thread state

    tstate->last_error = (cudaError_t)err;
    if (tstate->last_error) {
        return tstate->last_error;
    }

    CUcontext ctx_0;
    err = cuDevicePrimaryCtxRetain(&ctx_0, 0);
    tstate->last_error = (cudaError_t)err;
    if (tstate->last_error) {
        return tstate->last_error;
    }

    err = cuCtxSetCurrent(ctx_0);
    tstate->last_error = (cudaError_t)err;

    tstate->global = state;

    auto tstate_ptr = new std::shared_ptr<RuntimeThreadState>();
    *tstate_ptr = tstate;
    _runtime_thread_state.reset(tstate_ptr);

    return tstate->last_error;
}
