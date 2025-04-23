#include <fractos/core/controller_config.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/core/process.hpp>
#include <fractos/core/process_config.hpp>
#include <cstdlib>
#include <cuda.h>
#include <dlfcn.h>
#include <fractos/service/compute/cuda.hpp>
#include <glog/logging.h>
#include <lib.hpp>
#include <link.h>
#include <unordered_map>


// * FractOS objects

static std::mutex fractos_mutex;
static std::shared_ptr<fractos::core::process> process;
static std::shared_ptr<fractos::core::channel> channel;

std::string
get_env(std::string env_name, std::string default_str = "")
{
    auto res = default_str;
    auto env_val = secure_getenv(env_name.c_str());
    if (env_val) {
        res = env_val;
    }
    return res;
}

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


// * service object

static std::mutex state_mutex;
static std::shared_ptr<State> state;

State&
get_state()
{
    if (not state) [[unlikely]] {
        auto ch = get_channel_ptr();
        auto gns = fractos::core::gns::make_service();

        auto name = get_env("FRACTOS_SERVICE_COMPUTE_CUDA_NAME",
                            "fractos::service::compute::cuda");

        auto lock = std::unique_lock(state_mutex);
        if (not state) {
            state = std::make_shared<State>();
            state->service = fractos::service::compute::cuda::make_service(ch, *gns, name).get();
        }
    }

    DCHECK(state);
    return *state;
}

std::shared_ptr<fractos::service::compute::cuda::Device>
State::get_device_ordinal(int ordinal)
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
State::get_device(CUdevice device)
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
State::get_context(CUcontext context)
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
    auto& state = get_state();
    auto stack = new std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>>();
    state.context_stack.reset(stack);
    return stack;
}

std::stack<std::shared_ptr<fractos::service::compute::cuda::Context>> &
State::get_context_stack()
{
    auto ptr = context_stack.get();
    if (not ptr) [[unlikely]] {
        ptr = make_context_stack();
    }
    DCHECK(ptr);
    return *ptr;
}


// * symbol management

static void *libcuda_handle;

// auto-generated function map
struct cuda_function_t {
    char const* name;
    void* ptr;
};
extern "C" [[gnu::visibility("hidden")]] cuda_function_t default_functions[];

// NOTE: *cannot* be a global map, because it's constructed after init_lib() below
static std::unordered_map<std::string, void*> *implemented_functions;

// NOTE: not in cuda.h
extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus);

static decltype(&cuGetProcAddress_v2) ptr_cuGetProcAddress_v2;

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus)
{
    // LOG(INFO) << __func__ << "(" << symbol << ")";

    if (std::string(symbol) == "") {
        return (ptr_cuGetProcAddress_v2)(symbol, pfn, cudaVersion, flags, symbolStatus);
    }

    auto it = implemented_functions->find(symbol);
    if (it != implemented_functions->end()) {
        *pfn = it->second;
        return CUDA_SUCCESS;
    } else {
        *pfn = nullptr;
        LOG(WARNING) << "could not find function pointer for " << symbol;
        return CUDA_SUCCESS;
    }
}


// NOTE: cuda.h defines cuGetProcAddress -> cuGetProcAddress_v2
#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
{
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}

static decltype(&cuGetExportTable) ptr_cuGetExportTable;

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
{
    LOG(WARNING) << "TODO: properly handle cuGetExportTable";
    return (*ptr_cuGetExportTable)(ppExportTable, pExportTableId);
}


extern decltype(&cuInit) ptr_cuInit;

static void init_symbols() __attribute__((constructor));

static
void
init_symbols()
{
    std::string lib("libcuda.so");

    // allow env to override load path
    auto lib_env = secure_getenv("FRACTOS_SERVICE_COMPUTE_LIBCUDA");
    if (lib_env) {
        lib = lib_env;
    }

    libcuda_handle = dlopen(lib.c_str(), RTLD_LAZY);
    CHECK(libcuda_handle) << "--reason--> " << dlerror();

    char info[1024];
    CHECK(dlinfo(libcuda_handle, RTLD_DI_ORIGIN, info) == 0);
    LOG(INFO) << "opened backend cuda library in "
              << info << "/" << lib.substr(lib.find_last_of("/\\") + 1);

    implemented_functions = new std::unordered_map<std::string, void*>();

    for (size_t i = 0; true; i++) {
        auto& elem = default_functions[i];
        if (not elem.name) {
            break;
        }

        auto res = implemented_functions->insert(std::make_pair(elem.name, elem.ptr));
        CHECK(res.second) << "could not insert function pointer for " << elem.name;
    }


    auto do_load_sym = [&](auto name) {
        auto ptr = dlsym(libcuda_handle, name);
        CHECK(ptr) << "--reason--> " << dlerror();

        auto res = implemented_functions->insert(std::make_pair(name, ptr));
        CHECK(not res.second) << "could not insert function pointer for " << name;

        return ptr;
    };

#define load_sym_next(name) ptr_ ## name = (decltype(ptr_ ## name))do_load_sym(#name)

    // NOTE: globals override auto-generated weak symbols in default_functions
    load_sym_next(cuGetProcAddress_v2);
    load_sym_next(cuGetExportTable);
    load_sym_next(cuInit);

#undef load_sym_next
}
