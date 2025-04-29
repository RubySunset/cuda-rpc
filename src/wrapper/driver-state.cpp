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

std::mutex _driver_state_mutex;
std::atomic<std::shared_ptr<DriverState>> _driver_state;

std::shared_ptr<fractos::service::compute::cuda::Device>
DriverState::get_device_ordinal(int ordinal)
{
    {
        auto devices_lock = std::shared_lock(devices_mutex);
        auto it = ordinal_devices.find(ordinal);
        if (it != ordinal_devices.end()) {
            return it->second->device;
        }
    }

    {
        auto devices_lock = std::unique_lock(devices_mutex);

        auto it = ordinal_devices.find(ordinal);
        if (it != ordinal_devices.end()) {
            return it->second->device;
        }

        auto entry = std::make_shared<device_entry>();
        entry->device = service->device_get(ordinal).get();
        entry->context = nullptr;

        if (not entry->device) {
            return nullptr;
        }

        auto res1 = ordinal_devices.insert(std::make_pair(ordinal, entry));
        CHECK(res1.second);

        auto res2 = devices.insert(std::make_pair(entry->device->get_device(), entry));
        CHECK(res2.second);

        return entry->device;
    }
}

std::shared_ptr<DriverState::device_entry>
DriverState::_get_device_entry(CUdevice device)
{
    auto devices_lock = std::shared_lock(devices_mutex);
    auto it = devices.find(device);
    if (it != devices.end()) {
        DCHECK(it->second);
        return it->second;
    } else {
        return nullptr;
    }
}

std::shared_ptr<fractos::service::compute::cuda::Device>
DriverState::get_device(CUdevice device)
{
    auto entry = _get_device_entry(device);
    if (not entry) [[unlikely]] {
        return nullptr;
    }
    return entry->device;
}

std::shared_ptr<fractos::service::compute::cuda::Context>
DriverState::get_device_primary_context(CUdevice device)
{
    auto entry = _get_device_entry(device);
    if (not entry) [[unlikely]] {
        return nullptr;
    }
    auto ctx = entry->context.load(std::memory_order_acquire);
    if (not ctx) [[unlikely]] {
        auto devices_lock = std::unique_lock(devices_mutex);
        ctx = entry->context.load(std::memory_order_acquire);
        if (not ctx) {
            ctx = entry->device->make_context(0).get();
            entry->context = ctx;

            auto contexts_lock = std::unique_lock(contexts_mutex);
            auto res = contexts.insert(std::make_pair(ctx->get_context(), ctx));
            CHECK(res.second);
        }
    }
    return ctx;
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

std::shared_ptr<DriverState::module_desc>
DriverState::get_module(CUmodule module)
{
    auto lock = std::shared_lock(modules_mutex);
    auto it = modules.find(module);
    if (it != modules.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}
