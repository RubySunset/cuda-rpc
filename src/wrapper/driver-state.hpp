#pragma once

#include <boost/thread/tss.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/process.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <map>
#include <shared_mutex>
#include <stack>
#include <unordered_map>


fractos::core::process& get_process();
fractos::core::channel& get_channel();
std::shared_ptr<fractos::core::channel> get_channel_ptr();

struct DriverState {
    std::shared_ptr<fractos::service::compute::cuda::Service> service;

    // device

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

    // context

    std::shared_ptr<fractos::service::compute::cuda::Context> get_context(CUcontext ctx);
    void insert_context(std::shared_ptr<fractos::service::compute::cuda::Context> ctx);

    std::shared_ptr<fractos::service::compute::cuda::Context> get_current_context();
    std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>& get_context_stack();

private:
    std::shared_mutex contexts_mutex;
    std::unordered_map<CUcontext, std::shared_ptr<fractos::service::compute::cuda::Context>> contexts;
    std::unordered_map<CUcontext, std::shared_ptr<boost::thread_specific_ptr<std::shared_ptr<fractos::service::compute::cuda::Stream>>>> context_thread_streams;
public:

    boost::thread_specific_ptr<std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>> context_stack;

    // module

    struct module_desc {
        const void* image;
        size_t image_size;
        std::shared_ptr<fractos::service::compute::cuda::Module> module;
        std::mutex functions_mutex;
        std::unordered_map<std::string, CUfunction> functions;
    };

    std::shared_ptr<module_desc> get_module(CUmodule module);

    std::shared_mutex modules_mutex;
    std::unordered_map<CUmodule, std::shared_ptr<module_desc>> modules;

    // function

    struct func_desc {
        std::shared_ptr<fractos::service::compute::cuda::Function> function;
    };

    void insert_function(std::shared_ptr<func_desc> desc);
    void erase_function(std::shared_ptr<fractos::service::compute::cuda::Function> ptr);
    std::shared_ptr<func_desc> get_function(CUfunction function);

private:
    std::shared_mutex functions_mutex;
    std::unordered_map<CUfunction, std::shared_ptr<func_desc>> functions;
public:

    // library

    struct library_desc {
        std::shared_ptr<fractos::service::compute::cuda::Library> library;
    };

    void insert_library(std::shared_ptr<library_desc> library_desc);
    std::shared_ptr<library_desc> get_library(CUlibrary culibrary);

private:
    std::shared_mutex libraries_mutex;
    std::unordered_map<CUlibrary, std::shared_ptr<library_desc>> libraries;
public:

    // kernel

    struct kernel_desc {
        std::shared_ptr<fractos::service::compute::cuda::Kernel> kernel;
    };

    void insert_kernel(std::shared_ptr<kernel_desc> kernel_desc);
    std::shared_ptr<kernel_desc> get_kernel(CUkernel cukernel);

private:
    std::shared_mutex kernels_mutex;
    std::unordered_map<CUkernel, std::shared_ptr<kernel_desc>> kernels;
public:

    // memory

    std::shared_ptr<fractos::service::compute::cuda::Memory> get_memory(CUdeviceptr addr);
    void insert_memory(std::shared_ptr<fractos::service::compute::cuda::Memory> mem);
    std::shared_ptr<fractos::service::compute::cuda::Memory> erase_memory(CUdeviceptr addr);

private:
    std::shared_mutex mems_mutex;
    struct mem_desc {
        CUdeviceptr base;
        size_t size;
        std::shared_ptr<fractos::service::compute::cuda::Memory> mem;
    };
    std::map<CUdeviceptr, std::shared_ptr<mem_desc>> mems;
public:

    // stream

    std::shared_ptr<fractos::service::compute::cuda::Stream> get_stream(CUstream stream);
    std::shared_ptr<fractos::service::compute::cuda::Stream> get_stream_per_thread();

    std::shared_mutex streams_mutex;
    std::unordered_map<CUstream, std::shared_ptr<fractos::service::compute::cuda::Stream>> streams;

    // event

    std::shared_ptr<fractos::service::compute::cuda::Event> get_event(CUevent event);

    std::shared_mutex events_mutex;
    std::unordered_map<CUevent, std::shared_ptr<fractos::service::compute::cuda::Event>> events;
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
