#include <cuda.h>
#include <cuda_runtime.h>
#include <fatbinary_section.h>
#include <glog/logging.h>

#include "./runtime-state.hpp"


std::mutex& get_runtime_state_mutex()
{
    static std::mutex mutex;
    return mutex;
}

std::atomic<std::shared_ptr<RuntimeState>>& get_runtime_state_atomic()
{
    static std::atomic<std::shared_ptr<RuntimeState>> state;
    return state;
}

std::shared_ptr<RuntimeThreadState>& get_runtime_thread_state_ptr()
{
    thread_local std::shared_ptr<RuntimeThreadState> ptr;
    return ptr;
}


cudaError_t
do_runtime_init()
{
    CUresult err = CUDA_SUCCESS;

    auto tstate = std::make_shared<RuntimeThreadState>();

    auto state = get_runtime_state_atomic().load();
    if (state) {
        goto done_state;
    }

    // initialize global state

    {
        auto runtime_state_lock = std::unique_lock(get_runtime_state_mutex());

        state = get_runtime_state_atomic().load();
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
            err = cuDevicePrimaryCtxRetain(&ctx, dev);
            if (err != CUDA_SUCCESS) {
                goto err_state;
            }
        }

        get_runtime_state_atomic() = state;
    }

err_state:

done_state:

    // initialize thread state

    tstate->last_error = (cudaError_t)err;
    if (tstate->last_error) {
        return tstate->last_error;
    }

    CUcontext ctx_0;
    CUdevice dev_0;
    cuDeviceGet(&dev_0, 0);
    tstate->last_error = (cudaError_t)cuDevicePrimaryCtxRetain(&ctx_0, dev_0);
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
        get_runtime_thread_state_ptr() = tstate;
    }

    return tstate->last_error;
}

std::pair<cudaError_t, CUmodule>
RuntimeThreadState::get_module(const fatCubinHandle_t handle)
{
    LOG_FIRST_N(ERROR, 1) << "TODO: must return per-device CUmodule";

    auto error = cudaSuccess;
    auto lock = std::unique_lock(global->modules_mutex);

    auto it = global->fat_cubin_handles.find(handle);
    CHECK(it != global->fat_cubin_handles.end());

    auto module_desc = it->second;
    if (module_desc->module == 0) {
        auto desc = (const __fatBinC_Wrapper_t*)handle;
        CHECK(desc->magic == FATBINC_MAGIC);
        CHECK(desc->version == FATBINC_VERSION);

        CUmodule module = 0;
        error = (cudaError_t)cuModuleLoadData(&module, (const void*)desc->data);
        if (error == cudaSuccess) {
            module_desc->module = module;
        }
    }

    return std::make_pair(error, module_desc->module.load(std::memory_order_acquire));
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

        if (desc->modules.empty()) [[unlikely]] {
            CHECK(not desc->fat_cubin_handles.empty());
            for (auto& cubin_handle : desc->fat_cubin_handles) {
                auto [error, module] = get_module(cubin_handle);
                if (error != cudaSuccess) {
                    return std::make_pair(error, (CUfunction)0);
                }
                auto res = desc->modules.insert(module);
                CHECK(res.second);
            }
        }

        if (desc->func == 0) {
            CUfunction real_func;
            auto err = cuModuleGetFunction(&real_func, *desc->modules.begin(), desc->name.c_str());
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

        if (desc->module == 0) {
            auto [err, module] = get_module(desc->fat_cubin_handle);
            if (err != cudaSuccess) {
                return std::make_pair(err, (CUdeviceptr)nullptr);
            }

            desc->module = module;
        }

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
