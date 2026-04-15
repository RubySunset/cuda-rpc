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
#include <fractos/service/compute/cuda_msg.hpp>

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

// The representation of an event sync operation
struct EventSyncInfo {
    bool ready;
    std::shared_ptr<core::cap::request> cont;
};

// The possible tasks that can be processed by the cb manager
enum class TaskType {
    MEMCPY_ASYNC,
    STREAM_SYNC,
    CTX_SYNC,
    EVENT_SYNC,
};

// A task in the task queue
struct Task {
    TaskType task_type;
    unsigned long task_id;
};

// CUDA host functions used to notify the manager that a task is ready
inline void CUDA_CB notify_memcpy_ready(void*);
inline void CUDA_CB notify_stream_ready(void*);
inline void CUDA_CB notify_event_ready(void*);

// CUDA host function used to update number of remaining streams to wait for,
// when no streams remain, will notify memcpy manager like notify_XXX_ready
inline void CUDA_CB update_remaining_streams(void*);

class CudaHostCBManager {
public:
    ~CudaHostCBManager();

    void set_channel(std::shared_ptr<fractos::core::channel> ch);

    // Call once - will continuously process ready tasks until stop() is called
    void run();

    // Cause run() to exit
    void stop();

    // Enqueue an async memcpy operation on a stream
    CUresult enqueue_memcpy_async(CUcontext ctx, CUstream stream, std::shared_ptr<fractos::core::cap::memory> src, std::shared_ptr<fractos::core::cap::memory> dst);

    // Invoke the provided FractOS response when the stream is ready
    CUresult stream_sync(CUcontext ctx, CUstream stream, core::cap::request continuation);

    // Invoke the provided FractOS response when all streams are ready, in which case the context is ready
    CUresult ctx_sync(CUcontext ctx, std::vector<CUstream>& streams, core::cap::request continuation);

    CUresult event_create(CUevent event);
    CUresult event_record(CUstream stream, CUevent event);
    CUresult event_sync(CUevent event, core::cap::request continuation);
private:
    void process_ready_memcpy(unsigned int task_id);
    void process_ready_stream(unsigned int task_id);
    void process_ready_ctx(unsigned int task_id);
    void process_ready_event(unsigned int task_id);

    friend void CUDA_CB notify_memcpy_ready(void* userData);
    friend void CUDA_CB notify_stream_ready(void* userData);
    friend void CUDA_CB notify_event_ready(void* userData);
    friend void CUDA_CB update_remaining_streams(void* userData); 

    // Get an auxilliary stream that can be used to unblock memcpy streams
    // by calling cuStreamWriteValue
    CUstream get_aux_stream(CUcontext ctx);

    // General
    std::atomic<bool> stop_flag{false};
    std::atomic<unsigned long> next_task_id = 1; // start from 1 to avoid conflict with initial value of memcpy flag, which is 0
    std::mutex task_buf_mutex;
    std::condition_variable task_buf_cv;
    std::vector<Task> task_buf;
    std::shared_ptr<fractos::core::channel> ch;

    // Memcpy-specific
    std::mutex memcpy_info_mutex;
    std::unordered_map<unsigned long, MemcpyInfo> memcpy_info_map;
    std::unordered_map<CUcontext, std::unordered_map<CUstream, std::pair<CUdeviceptr, CUdeviceptr>>> flag_map;
        // pair.first is used to temporarily block the stream to allow all CUDA operations to be enqueued without starting
        // execution, when calling memcpy
        // pair.second is used to block the stream while the channel copy itself is executing
    std::unordered_map<CUcontext, CUstream> aux_stream_map;

    // Stream sync-specific
    std::mutex stream_cont_mutex;
    std::unordered_map<unsigned long, core::cap::request> stream_cont_map;

    // Context sync-specific
    std::mutex ctx_info_mutex;
    std::unordered_map<unsigned long, core::cap::request> ctx_cont_map;
    std::unordered_map<unsigned long, std::atomic<unsigned>> num_remaining_streams_map;

    // Event sync-specific
    std::mutex event_mutex;
    std::unordered_map<CUevent, unsigned long> event_to_id;
    std::unordered_map<unsigned long, CUevent> id_to_event;
    std::unordered_map<unsigned long, EventSyncInfo> event_info;
};

inline
CudaHostCBManager&
get_cuda_host_cb_manager()
{
    static CudaHostCBManager cuda_host_cb_manager;
    return cuda_host_cb_manager;
}
