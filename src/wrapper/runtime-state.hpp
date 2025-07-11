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


struct RuntimeState {
    struct func_desc {
        std::unordered_set<CUmodule> modules;
        std::string name;
        std::atomic<CUfunction> func;
        std::mutex mutex;
    };

    struct var_desc {
        CUmodule module;
        std::string name;
        std::atomic<CUdeviceptr> address;
        std::mutex mutex;
    };

    struct module_desc {
        std::mutex entries_mutex;
        std::unordered_set<uintptr_t> funcs;
        std::unordered_set<uintptr_t> vars;
    };

    std::mutex modules_mutex;
    std::unordered_map<CUmodule, std::shared_ptr<module_desc>> modules;

    std::shared_mutex entries_mutex;
    std::unordered_map<uintptr_t, std::shared_ptr<func_desc>> funcs;
    std::unordered_map<uintptr_t, std::shared_ptr<var_desc>> vars;
};

struct RuntimeThreadState {
    cudaError_t last_error;
    int dev_o;
    CUdevice dev;
    std::shared_ptr<RuntimeState> global;

    std::pair<cudaError_t, CUfunction> get_function(const void* address);
    std::pair<cudaError_t, CUdeviceptr> get_variable(const void* address);
};

extern std::mutex _runtime_state_mutex;
extern std::atomic<std::shared_ptr<RuntimeState>> _runtime_state;
extern boost::thread_specific_ptr<std::shared_ptr<RuntimeThreadState>> _runtime_thread_state;

cudaError_t do_runtime_init();

#define get_runtime_state_with_error()                                  \
    ({                                                                  \
        cudaError_t err = cudaSuccess;                                  \
        std::shared_ptr<RuntimeThreadState> state;                      \
        auto tstate = _runtime_thread_state.get();                      \
        if (not tstate) [[unlikely]] {                                  \
            err = do_runtime_init();                                    \
            if (err == cudaSuccess) {                                   \
                DCHECK(_runtime_thread_state.get());                    \
                state = *_runtime_thread_state.get();                   \
            }                                                           \
        } else {                                                        \
            state = *tstate;                                            \
        }                                                               \
        std::make_pair(err, state);                                     \
    })

#define get_runtime_state()                                             \
    ({                                                                  \
        auto tstate = _runtime_thread_state.get();                      \
        if (not tstate) [[unlikely]] {                                  \
            auto err = do_runtime_init();                               \
            if (err != cudaSuccess) {                                   \
                return err;                                             \
            }                                                           \
            tstate = _runtime_thread_state.get();                       \
        }                                                               \
        DCHECK(tstate);                                                 \
        std::ref(**tstate);                                             \
    }).get()

#define get_runtime_state_unsafe()                                      \
    ({                                                                  \
        auto tstate = _runtime_thread_state.get();                      \
        DCHECK(tstate);                                                 \
        DCHECK(*tstate);                                                \
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
