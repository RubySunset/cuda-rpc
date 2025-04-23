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

struct State {
    std::shared_ptr<fractos::service::compute::cuda::Service> service;


    std::shared_ptr<fractos::service::compute::cuda::Device> get_device_ordinal(int ordinal);
    std::shared_ptr<fractos::service::compute::cuda::Device> get_device(CUdevice device);

    std::shared_mutex devices_mutex;
    std::unordered_map<int, std::shared_ptr<fractos::service::compute::cuda::Device>> ordinal_devices;
    std::unordered_map<CUdevice, std::shared_ptr<fractos::service::compute::cuda::Device>> devices;


    std::shared_ptr<fractos::service::compute::cuda::Context> get_context(CUcontext ctx);

    std::shared_mutex contexts_mutex;
    std::unordered_map<CUcontext, std::shared_ptr<fractos::service::compute::cuda::Context>> contexts;


    std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>& get_context_stack();

    boost::thread_specific_ptr<std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>> context_stack;
};

State& get_state();
