#pragma once

#include <boost/thread/tss.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/process.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <shared_mutex>
#include <stack>
#include <unordered_map>


fractos::core::process& get_process();
fractos::core::channel& get_channel();
std::shared_ptr<fractos::core::channel> get_channel_ptr();

struct DriverState {
    std::shared_ptr<fractos::service::compute::cuda::Service> service;


    std::shared_ptr<fractos::service::compute::cuda::Device> get_device_ordinal(int ordinal);
    std::shared_ptr<fractos::service::compute::cuda::Device> get_device(CUdevice device);
    std::shared_ptr<fractos::service::compute::cuda::Context> get_device_primary_context(CUdevice device);

    struct device_entry {
        std::shared_ptr<fractos::service::compute::cuda::Device> device;
        std::atomic<std::shared_ptr<fractos::service::compute::cuda::Context>> context;
    };

    std::shared_mutex devices_mutex;
    std::unordered_map<int, std::shared_ptr<device_entry>> ordinal_devices;
    std::unordered_map<CUdevice, std::shared_ptr<device_entry>> devices;
private:
    std::shared_ptr<device_entry> _get_device_entry(CUdevice device);
public:


    std::shared_ptr<fractos::service::compute::cuda::Context> get_context(CUcontext ctx);

    std::shared_mutex contexts_mutex;
    std::unordered_map<CUcontext, std::shared_ptr<fractos::service::compute::cuda::Context>> contexts;


    std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>& get_context_stack();

    boost::thread_specific_ptr<std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>> context_stack;
};

extern std::mutex _driver_state_mutex;
extern std::atomic<std::shared_ptr<DriverState>> _driver_state;

#define get_driver_state()                                              \
    ({                                                                  \
        auto state = _driver_state.load(std::memory_order_consume);     \
        if (not state) [[unlikely]] {                                   \
            return CUDA_ERROR_NOT_INITIALIZED;                          \
        }                                                               \
        std::ref(*state);                                               \
    }).get()

#define get_driver_state_unsafe()                                       \
    ({                                                                  \
        auto state = _driver_state.load(std::memory_order_consume);     \
        DCHECK(state);                                                  \
        std::ref(*state);                                               \
    }).get()
