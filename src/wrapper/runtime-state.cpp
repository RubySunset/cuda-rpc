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
    tstate->last_error = (cudaError_t)cuDevicePrimaryCtxRetain(&ctx_0, 0);
    if (tstate->last_error) {
        return tstate->last_error;
    }

    err = cuCtxSetCurrent(ctx_0);
    tstate->last_error = (cudaError_t)err;

    tstate->global = state;
    tstate->dev_o = 0;
    tstate->last_error = (cudaError_t)cuDeviceGet(&tstate->dev, tstate->dev_o);
    if (tstate->last_error) {
        return tstate->last_error;
    }

    {
        auto tstate_ptr = new std::shared_ptr<RuntimeThreadState>();
        *tstate_ptr = tstate;
        _runtime_thread_state.reset(tstate_ptr);
    }

    return tstate->last_error;
}

std::pair<cudaError_t, CUfunction>
RuntimeThreadState::get_function(const void* address)
{
    LOG_FIRST_N(ERROR, 1) << "TODO: must return per-device CUfunction";

    auto lock = std::shared_lock(global->entries_mutex);
    auto it = global->funcs.find((uintptr_t)address);
    if (it == global->funcs.end()) [[unlikely]] {
        return std::make_pair(cudaErrorInvalidDeviceFunction, nullptr);
    }

    auto desc = it->second;
    auto func = desc->func.load(std::memory_order_acquire);

    if (func == 0) [[unlikely]] {
        auto lock = std::unique_lock(desc->mutex);
        if (desc->func == 0) {
            CUfunction real_func;
            auto err = cuModuleGetFunction(&real_func, desc->module, desc->name.c_str());
            if (err != CUDA_SUCCESS) {
                return std::make_pair((cudaError_t)err, nullptr);
            }
            // NOTE: std::atomic<T*> zeroes source
            desc->func.store(real_func);
        }
        func = desc->func.load();
    }

    return std::make_pair(cudaSuccess, func);
}

std::pair<cudaError_t, CUdeviceptr>
RuntimeThreadState::get_variable(const void* address)
{
    LOG_FIRST_N(ERROR, 1) << "TODO: must return per-device CUdeviceptr";

    auto lock = std::shared_lock(global->entries_mutex);
    auto it = global->vars.find((uintptr_t)address);
    if (it == global->vars.end()) [[unlikely]] {
        return std::make_pair(cudaErrorInvalidSymbol, (CUdeviceptr)nullptr);
    }

    auto desc = it->second;
    auto cuaddr = desc->address.load(std::memory_order_acquire);

    if (cuaddr == 0) [[unlikely]] {
        auto lock = std::unique_lock(desc->mutex);
        if (desc->address == 0) {
            CUdeviceptr real_cuaddr;
            auto err = cuModuleGetGlobal(&real_cuaddr, nullptr, desc->module, desc->name.c_str());
            if (err != CUDA_SUCCESS) {
                return std::make_pair((cudaError_t)err, (CUdeviceptr)nullptr);
            }
            // NOTE: std::atomic<T*> zeroes source
            desc->address.store(real_cuaddr);
        }
        cuaddr = desc->address.load();
    }

    return std::make_pair(cudaSuccess, cuaddr);
}
