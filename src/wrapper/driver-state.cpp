#include <fractos/core/controller_config.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/core/process.hpp>
#include <fractos/core/process_config.hpp>

#include <./common.hpp>
#include <./driver-state.hpp>


// * FractOS objects

static std::mutex fractos_mutex;
static std::shared_ptr<fractos::core::process> process;
static std::shared_ptr<fractos::core::channel> channel;

fractos::core::process&
get_process()
{
    if (not process) [[unlikely]] {
        auto controller_conf = fractos::core::parse_controller_config(
            get_env("FRACTOS_SERVICE_COMPUTE_CUDA_CONTROLLER"));
        auto process_conf = fractos::core::parse_process_config(
            get_env("FRACTOS_SERVICE_COMPUTE_CUDA_PROCESS"));

        auto lock = std::unique_lock(fractos_mutex);
        if (not process) {
            process = fractos::core::make_process(controller_conf, process_conf).get();
        }
    }

    DCHECK(process);
    return *process;
}

std::shared_ptr<fractos::core::channel>
get_channel_ptr()
{
    if (not channel) [[unlikely]] {
        auto process = get_process();
        auto channel_conf = fractos::core::parse_channel_config(
            get_env("FRACTOS_SERVICE_COMPUTE_CUDA_CHANNEL"));

        auto lock = std::unique_lock(fractos_mutex);
        if (not channel) {
            channel = process.make_channel(channel_conf).get();
        }
    }

    DCHECK(channel);
    return channel;
}

fractos::core::channel&
get_channel()
{
    return *get_channel_ptr();
}


// * DriverState object

std::shared_ptr<fractos::service::compute::cuda::Device>
DriverState::get_device_ordinal(int ordinal)
{
    {
        auto devices_lock = std::shared_lock(devices_mutex);
        auto it = ordinal_devices.find(ordinal);
        if (it != ordinal_devices.end()) {
            return it->second;
        }
    }

    {
        auto devices_lock = std::unique_lock(devices_mutex);

        auto it = ordinal_devices.find(ordinal);
        if (it != ordinal_devices.end()) {
            return it->second;
        }

        auto device_ptr = service->device_get(ordinal).get();

        auto res1 = ordinal_devices.insert(std::make_pair(ordinal, device_ptr));
        CHECK(res1.second);

        auto res2 = devices.insert(std::make_pair(device_ptr->get_device(), device_ptr));
        CHECK(res2.second);

        return device_ptr;
    }
}

std::shared_ptr<fractos::service::compute::cuda::Device>
DriverState::get_device(CUdevice device)
{
    auto devices_lock = std::shared_lock(devices_mutex);
    auto it = devices.find(device);
    if (it != devices.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

std::shared_ptr<fractos::service::compute::cuda::Context>
DriverState::get_context(CUcontext context)
{
    auto contexts_lock = std::shared_lock(contexts_mutex);
    auto it = contexts.find(context);
    if (it != contexts.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

static
auto
make_context_stack()
{
    auto& state = get_driver_state_unsafe();
    auto stack = new std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>();
    state.context_stack.reset(stack);
    return stack;
}

std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>> &
DriverState::get_context_stack()
{
    auto ptr = context_stack.get();
    if (not ptr) [[unlikely]] {
        ptr = make_context_stack();
    }
    DCHECK(ptr);
    return *ptr;
}
