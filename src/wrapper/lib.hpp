#pragma once

#include <fractos/core/channel.hpp>
#include <fractos/core/process.hpp>
#include <fractos/service/compute/cuda.hpp>


fractos::core::process& get_process();
fractos::core::channel& get_channel();
std::shared_ptr<fractos::core::channel> get_channel_ptr();

struct State {
    std::shared_ptr<fractos::service::compute::cuda::Service> service;

    std::mutex devices_mutex;
    std::list<std::shared_ptr<fractos::service::compute::cuda::Device>> devices;
};

State& get_state();
