#pragma once

#include <atomic>
#include <boost/thread/tss.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>


using fatCubinHandle_t = void*;

struct RuntimeState {
    struct func_desc {
        std::unordered_set<fatCubinHandle_t> fat_cubin_handles;
        std::unordered_set<CUmodule> modules;
        std::string name;
        std::atomic<CUfunction> func;
        std::mutex mutex;
    };

    struct var_desc {
        fatCubinHandle_t fat_cubin_handle;
        std::atomic<CUmodule> module;
        std::string name;
        std::atomic<CUdeviceptr> address;
        std::mutex mutex;
    };

    struct module_desc {
        std::mutex entries_mutex;
        fatCubinHandle_t fat_cubin_handle;
        std::atomic<CUmodule> module;
        std::unordered_set<uintptr_t> funcs;
        std::unordered_set<uintptr_t> vars;
    };

    std::mutex modules_mutex;
    std::unordered_map<fatCubinHandle_t, std::shared_ptr<module_desc>> fat_cubin_handles;
    std::unordered_map<CUmodule, std::shared_ptr<module_desc>> modules;

    std::shared_mutex entries_mutex;
    std::unordered_map<uintptr_t, std::shared_ptr<func_desc>> funcs;
    std::unordered_map<uintptr_t, std::shared_ptr<var_desc>> vars;

    std::mutex kernel_mutex;
    uintptr_t kernel_cnt = 0;
    std::unordered_map<cudaKernel_t, const void*> kernel_to_addr;
    std::unordered_map<const void*, cudaKernel_t> addr_to_kernel;
};

struct RuntimeThreadState {
    cudaError_t last_error;
    int dev_o;
    CUdevice dev;
    std::shared_ptr<RuntimeState> global;

    std::pair<cudaError_t, CUmodule> get_module(fatCubinHandle_t handle);
    std::pair<cudaError_t, CUfunction> get_function(const void* address);
    std::pair<cudaError_t, CUdeviceptr> get_variable(const void* address);
};

std::mutex& get_runtime_state_mutex();
std::atomic<std::shared_ptr<RuntimeState>>& get_runtime_state_storage();
std::shared_ptr<RuntimeThreadState>& get_runtime_thread_state_ptr();

cudaError_t do_runtime_init();

#define get_runtime_state_with_error()                                  \
    ({                                                                  \
        cudaError_t err = cudaSuccess;                                  \
        std::shared_ptr<RuntimeThreadState> state;                      \
        auto tstate = get_runtime_thread_state_ptr();                   \
        if (not tstate) [[unlikely]] {                                  \
            err = do_runtime_init();                                    \
            if (err == cudaSuccess) {                                   \
                DCHECK(get_runtime_thread_state_ptr());                 \
                state = get_runtime_thread_state_ptr();                \
            }                                                           \
        } else {                                                        \
            state = tstate;                                            \
        }                                                               \
        std::make_pair(err, state);                                     \
    })

#define get_runtime_state()                                             \
    ({                                                                  \
        auto tstate = get_runtime_thread_state_ptr();                   \
        if (not tstate) [[unlikely]] {                                  \
            auto err = do_runtime_init();                               \
            if (err != cudaSuccess) {                                   \
                return err;                                             \
            }                                                           \
            tstate = get_runtime_thread_state_ptr();                    \
        }                                                               \
        DCHECK(tstate);                                                 \
        std::ref(*tstate);                                             \
    }).get()

#define get_runtime_state_unsafe()                                      \
    ({                                                                  \
        auto tstate = get_runtime_thread_state_ptr();                   \
        DCHECK(tstate);                                                 \
        std::ref(*tstate);                                              \
    }).get()

#define return_error(err)                       \
    state.last_error = err;                     \
    return err;                                 \

#define return_error_maybe(err)                 \
    state.last_error = err;                     \
    if (err) {                                  \
        return err;                             \
    }
