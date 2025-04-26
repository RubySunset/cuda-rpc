#pragma once

#include <atomic>
#include <boost/thread/tss.hpp>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>


struct RuntimeState {
};

struct RuntimeThreadState {
    std::shared_ptr<RuntimeState> global;
};

extern std::mutex _runtime_state_mutex;
extern std::atomic<std::shared_ptr<RuntimeState>> _runtime_state;
extern boost::thread_specific_ptr<std::shared_ptr<RuntimeThreadState>> _runtime_thread_state;

cudaError_t do_runtime_init();

#define get_runtime_state_with_error()                                  \
    ({                                                                  \
        cudaError_t err;                                                \
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
            if (err == cudaSuccess) {                                   \
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
