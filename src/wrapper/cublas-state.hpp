#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <glog/logging.h>

#include <fractos/service/compute/cuda.hpp>

#include "./driver-state.hpp"
#include "./runtime-state.hpp"


namespace srv = fractos::service::compute::cuda;


class CublasState {
public:
    void add_handle(cublasHandle_t handle, std::shared_ptr<srv::CublasHandle> cublas_obj);

    std::shared_ptr<srv::CublasHandle> get_handle(cublasHandle_t handle);

    bool erase_handle(cublasHandle_t handle);

    // Update the stream associated with a cublas handle
    // (normally done via cublasSetStream)
    void update_stream(cublasHandle_t handle, std::shared_ptr<srv::Stream> stream_obj);

    // Get the stream associated with a cublas handle
    // (normally done via cublasGetStream)
    std::shared_ptr<srv::Stream> get_stream(cublasHandle_t handle);

    bool erase_stream(cublasHandle_t handle);
private:
    std::mutex mut;
    std::unordered_map<cublasHandle_t, std::shared_ptr<srv::CublasHandle>> cublas_handle_map;
    std::unordered_map<cublasHandle_t, std::shared_ptr<srv::Stream>> stream_map;
};


inline CublasState& get_cublas_state_instance() {
    static CublasState instance;
    return instance;
}


#define get_driver_state_return_cublas()                                \
    ({                                                                  \
        auto state = get_driver_state_ptr().load(); \
        if (not state) [[unlikely]] {                                   \
            return CUBLAS_STATUS_NOT_INITIALIZED;                       \
        }                                                               \
        std::ref(*state);                                               \
    }).get()

#define get_runtime_state_return_cublas()                               \
    ({                                                                  \
        auto tstate = get_runtime_thread_state_ptr();                   \
        if (not tstate) [[unlikely]] {                                  \
            auto err = do_runtime_init();                               \
            if (err != cudaSuccess) {                                   \
                return CUBLAS_STATUS_NOT_INITIALIZED;                   \
            }                                                           \
            tstate = get_runtime_thread_state_ptr();                    \
        }                                                               \
        DCHECK(tstate);                                                 \
        std::ref(*tstate);                                             \
    }).get()

#define get_cublas_state()                          \
    ({                                              \
        get_runtime_state_return_cublas();          \
        std::ref(get_cublas_state_instance());      \
    }).get()
