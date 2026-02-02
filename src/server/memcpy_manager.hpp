#pragma once

#include <cstring>
#include <condition_variable>
#include <memory>
#include <vector>
#include <mutex>
#include <unordered_map>

#include <cuda.h>

#include <fractos/core/channel.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/common/logging.hpp>

#include "./common.hpp"

using namespace fractos;


// The representation of an async memcpy operation that has not yet started
struct MemcpyInfo {
    std::shared_ptr<fractos::core::cap::memory> src;
    std::shared_ptr<fractos::core::cap::memory> dst;
    CUcontext ctx;
    CUstream stream;
};

// A CUDA host function used to trigger the start of an async memcpy
inline void CUDA_CB notify_memcpy_ready(void*);

class MemcpyManager {
public:
    ~MemcpyManager();

    void set_channel(std::shared_ptr<fractos::core::channel> ch);

    // Call once - will continuously process pending memcpy ops when ready
    // until stop() is called
    void run();

    // Cause run() to exit
    void stop();

    // Queue an async memcpy operation on a stream
    void enqueue_memcpy_async(CUcontext ctx, CUstream stream, std::shared_ptr<fractos::core::cap::memory> src, std::shared_ptr<fractos::core::cap::memory> dst);
private:
    friend void CUDA_CB notify_memcpy_ready(void* userData);

    // Get an auxilliary stream that can be used to unblock memcpy streams
    // by calling cuStreamWriteValue
    CUstream get_aux_stream(CUcontext ctx);

    std::atomic<bool> stop_flag{false};
    std::mutex memcpy_info_mutex;
    unsigned long next_memcpy_id = 1; // start from 1 to avoid conflict with initial value of flag, which is 0
    std::unordered_map<unsigned long, MemcpyInfo> memcpy_info_map;
    std::mutex memcpy_buf_mutex;
    std::vector<unsigned long> memcpy_buf;
    std::condition_variable memcpy_buf_cv;
    std::shared_ptr<fractos::core::channel> ch;
    std::unordered_map<CUcontext, std::unordered_map<CUstream, CUdeviceptr>> flag_map;
    std::unordered_map<CUcontext, CUstream> aux_stream_map;
};

inline MemcpyManager& get_memcpy_manager() {
    static MemcpyManager memcpy_manager;
    return memcpy_manager;
}
