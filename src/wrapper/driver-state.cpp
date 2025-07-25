#include <fractos/core/controller_config.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/core/process.hpp>
#include <fractos/core/process_config.hpp>
#include <fractos/service/compute/cuda.hpp>

#include <./common.hpp>
#include <./driver-state.hpp>


namespace clt = fractos::service::compute::cuda;


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

            insert_context(ctx);
        }
    }
    return ctx;
}

void
DriverState::insert_context(std::shared_ptr<fractos::service::compute::cuda::Context> ctx)
{
    auto contexts_lock = std::unique_lock(contexts_mutex);

    auto res1 = contexts.insert(std::make_pair(ctx->get_context(), ctx));
    CHECK(res1.second);

    auto tptr = std::make_shared<
        boost::thread_specific_ptr<std::shared_ptr<fractos::service::compute::cuda::Stream>>>();
    auto res2 = context_thread_streams.emplace(std::make_pair(ctx->get_context(), tptr));
    CHECK(res2.second);
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

std::shared_ptr<fractos::service::compute::cuda::Context>
DriverState::get_current_context()
{
    auto& stack = get_context_stack();
    if (not stack.empty()) {
        return stack.top();
    } else {
        return nullptr;
    }
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


void
DriverState::insert_function(std::shared_ptr<DriverState::func_desc> desc)
{
    auto lock = std::unique_lock(functions_mutex);
    auto res = functions.insert(std::make_pair(desc->function->get_function(), desc));
    CHECK(res.second);
}

void
DriverState::erase_function(std::shared_ptr<fractos::service::compute::cuda::Function> ptr)
{
    auto lock = std::unique_lock(functions_mutex);
    auto res = functions.erase(ptr->get_function());
    LOG_IF(ERROR, res != 1) << "could not find " << fractos::service::compute::cuda::to_string(*ptr);
}

std::shared_ptr<DriverState::func_desc>
DriverState::get_function(CUfunction function)
{
    auto lock = std::shared_lock(functions_mutex);
    auto it = functions.find(function);
    if (it != functions.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}


void
DriverState::insert_library(std::shared_ptr<DriverState::library_desc> desc)
{
    auto lock = std::unique_lock(libraries_mutex);
    auto res = libraries.insert(std::make_pair(desc->library->get_library(), desc));
    CHECK(res.second);
}

std::shared_ptr<DriverState::library_desc>
DriverState::get_library(CUlibrary library)
{
    auto lock = std::shared_lock(libraries_mutex);
    auto it = libraries.find(library);
    if (it != libraries.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}


void
DriverState::insert_kernel(std::shared_ptr<DriverState::kernel_desc> desc)
{
    auto lock = std::unique_lock(kernels_mutex);
    auto res = kernels.insert(std::make_pair(desc->kernel->get_kernel(), desc));
    CHECK(res.second);
}


std::shared_ptr<clt::Memory>
DriverState::get_memory(CUdeviceptr addr)
{
    auto lock = std::shared_lock(mems_mutex);
    auto it = mems.upper_bound(addr);
    if (it == mems.end()) {
        return nullptr;
    } else if (it->second->base <= addr){
        CHECK(addr < (it->second->base + it->second->size));
        return it->second->mem;
    } else {
        return nullptr;
    }
}

void
DriverState::insert_memory(std::shared_ptr<clt::Memory> mem)
{
    auto& cap = mem->get_cap_mem();
    DCHECK(cap.get_addr() == mem->get_deviceptr());
    auto mem_desc = std::make_shared<DriverState::mem_desc>();
    mem_desc->base = cap.get_addr();
    mem_desc->size = cap.get_size();
    mem_desc->mem = mem;

    auto lock = std::unique_lock(mems_mutex);
    auto res = mems.insert(std::make_pair(mem_desc->base+mem_desc->size, mem_desc));
    CHECK(res.second);
}

std::shared_ptr<clt::Memory>
DriverState::erase_memory(CUdeviceptr addr)
{
    auto lock = std::unique_lock(mems_mutex);
    auto it = mems.find(addr);
    if (it == mems.end()) {
        return nullptr;
    } else if (it->second->base <= addr and addr < (it->second->base + it->second->size)){
        auto res = it->second;
        mems.erase(it);
        return res->mem;
    } else {
        return nullptr;
    }
}

std::shared_ptr<fractos::service::compute::cuda::Stream>
DriverState::get_stream(CUstream stream)
{
    if (stream == CU_STREAM_PER_THREAD) {
        return get_stream_per_thread();
    }

    auto lock = std::shared_lock(streams_mutex);
    auto it = streams.find(stream);
    if (it != streams.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}

std::shared_ptr<fractos::service::compute::cuda::Stream>
DriverState::get_stream_per_thread()
{
    auto ctx = get_current_context();
    if (not ctx) {
        return nullptr;
    }

    std::shared_ptr<boost::thread_specific_ptr<std::shared_ptr<fractos::service::compute::cuda::Stream>>> tptr;
    {
        auto lock = std::shared_lock(contexts_mutex);
        auto it = context_thread_streams.find(ctx->get_context());
        CHECK(it != context_thread_streams.end());

        tptr = it->second;
    }

    auto* ptr = tptr->get();
    if (ptr) {
        return *ptr;
    } else {
        auto stream = new std::shared_ptr<fractos::service::compute::cuda::Stream>(
            ctx->stream_create(CU_STREAM_DEFAULT).get());
        tptr->reset(stream);
        return *stream;
    }
}

std::shared_ptr<fractos::service::compute::cuda::Event>
DriverState::get_event(CUevent event)
{
    auto lock = std::shared_lock(events_mutex);
    auto it = events.find(event);
    if (it != events.end()) {
        return it->second;
    } else {
        return nullptr;
    }
}
