#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fatbinary_section.h>
#include <fractos/logging.hpp>
#include <glog/logging.h>
#include <string>
#include <unordered_map>

#include "./runtime-state.hpp"
#include "./runtime-syms-extern.hpp"

#define SYM(name) decltype(&name) ptr_ ## name;
#include "./runtime-syms.hpp"
#undef SYM

using namespace fractos;


static void *libcudart_handle;

// auto-generated function map
struct cuda_function_t {
    char const* name;
    void* ptr;
};
extern "C" [[gnu::visibility("hidden")]] cuda_function_t runtime_default_functions[];

// NOTE: *cannot* be a global map, because it's constructed after init_lib() below
static std::unordered_map<std::string, void*> *implemented_functions;


extern "C" [[gnu::visibility("default")]]
void** CUDARTAPI
__cudaRegisterFatBinary(void* fatCubin)
{
    CUmodule module;

    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::APP)
            << "__cudaRegisterFatBinary ->"
            << " fatCubin=0x" << std::hex << fatCubin;
        module = 0;
        DVLOG(logging::APP)
            << "__cudaRegisterFatBinary <- "
            << "0x" << std::hex << module;
        return (void**)module;
    }

    CHECK(state_ptr);
    auto& state [[maybe_unused]] = *state_ptr;

    auto desc = (const __fatBinC_Wrapper_t*)fatCubin;
    CHECK(desc->magic == FATBINC_MAGIC);
    CHECK(desc->version == FATBINC_VERSION);

    DVLOG(logging::APP)
        << "__cudaRegisterFatBinary ->"
        << " fatCubin=0x" << std::hex << fatCubin
        << " desc->data=0x" << std::hex << desc->data;

    state.last_error = (cudaError_t)cuModuleLoadData(&module, (const void*)desc->data);
    if (state.last_error != cudaSuccess) {
        LOG(FATAL) << "failed cuModuleLoadData(..., " << (const void*)desc->data << "): "
                   << cudaGetErrorName(state.last_error);
    }

    auto module_desc = std::make_shared<RuntimeState::module_desc>();
    {
        auto modules_lock = std::unique_lock(state.global->modules_mutex);
        auto res = state.global->modules.insert(std::make_pair(module, module_desc));
        CHECK(res.second);
    }

    DVLOG(logging::APP)
        << "__cudaRegisterFatBinary <- "
        << "0x" << std::hex << module;

    return (void**)module;
}

extern "C" [[gnu::visibility("default")]]
void CUDARTAPI
__cudaRegisterFatBinaryEnd(void** fatCubinHandle)
{
    if (not fatCubinHandle) {
        return;
    }

    DVLOG(logging::APP)
        << "__cudaRegisterFatBinaryEnd ->"
        << " fatCubinHandle=0x" << std::hex << fatCubinHandle;
}

extern "C" [[gnu::visibility("default")]]
void CUDARTAPI
__cudaUnregisterFatBinary(void** fatCubinHandle)
{
    if (not fatCubinHandle) {
        return;
    }

    DVLOG(logging::APP)
        << "__cudaUnregisterFatBinary ->"
        << " fatCubinHandle=0x" << std::hex << fatCubinHandle;

    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::APP)
            << "__cudaUnregisterFatBinary <- "
            << " err=" << cudaGetErrorName((cudaError_t)err);
        return;
    }
    auto& state = *state_ptr;

    auto module = (CUmodule)fatCubinHandle;

    auto modules_lock = std::unique_lock(state.global->modules_mutex);
    auto entries_lock = std::unique_lock(state.global->entries_mutex);

    state.last_error = (cudaError_t)cuModuleUnload(module);
    if (state.last_error != cudaSuccess) {
        goto out;
    }

    {
        auto module_desc_it = state.global->modules.find(module);
        CHECK(module_desc_it != state.global->modules.end());
        auto module_desc = module_desc_it->second;

        auto module_entries_lock = std::unique_lock(module_desc->entries_mutex);
        for (auto& func : module_desc->funcs) {
            CHECK(state.global->funcs.erase(func) == 1);
        }
        for (auto& func : module_desc->vars) {
            CHECK(state.global->vars.erase(func) == 1);
        }

        state.global->modules.erase(module_desc_it);
    }

out:
    DVLOG(logging::APP)
        << "__cudaUnregisterFatBinary <-"
        << " err=" << cudaGetErrorName(state.last_error);
}

extern "C" [[gnu::visibility("default")]]
void CUDARTAPI __cudaRegisterFunction(
        void   **fatCubinHandle,
  const char    *hostFun,
        char    *deviceFun,
  const char    *deviceName,
        int      thread_limit,
        uint3   *tid,
        uint3   *bid,
        dim3    *bDim,
        dim3    *gDim,
        int     *wSize
    )
{
    DVLOG(logging::APP)
        << "__cudaRegisterFunction ->"
        << " handle=" << fatCubinHandle
        // << " hostFun=" << (void*)hostFun
        << " deviceName=" << deviceName;

    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::APP)
            << "__cudaRegisterFunction <- "
            << " err=" << cudaGetErrorName((cudaError_t)err);
        return;
    }
    auto& state = *state_ptr;

    auto module = (CUmodule)fatCubinHandle;

    auto modules_lock = std::unique_lock(state.global->modules_mutex);
    auto module_desc_it = state.global->modules.find(module);
    CHECK(module_desc_it != state.global->modules.end());
    auto module_desc = module_desc_it->second;

    auto func_desc = std::make_shared<RuntimeState::func_desc>();
    func_desc->module = module;
    func_desc->name = deviceName;
    func_desc->func = 0;

    auto module_entries_lock = std::unique_lock(module_desc->entries_mutex);
    CHECK(module_desc->funcs.insert((uintptr_t)hostFun).second);

    auto entries_lock = std::unique_lock(state.global->entries_mutex);
    CHECK(state.global->funcs.insert(std::make_pair((uintptr_t)hostFun, func_desc)).second);
}

extern "C" [[gnu::visibility("default")]]
void CUDARTAPI __cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
    )
{
    DVLOG(logging::APP)
        << "__cudaRegisterVar ->"
        << " handle=" << fatCubinHandle
        << " hostVar=" << (void*)hostVar
        << " deviceName=" << deviceName;

    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::APP)
            << "__cudaRegisterVar <- "
            << " err=" << cudaGetErrorName((cudaError_t)err);
        return;
    }
    auto& state = *state_ptr;

    auto module = (CUmodule)fatCubinHandle;

    auto modules_lock = std::unique_lock(state.global->modules_mutex);
    auto module_desc_it = state.global->modules.find(module);
    CHECK(module_desc_it != state.global->modules.end());
    auto module_desc = module_desc_it->second;

    auto var_desc = std::make_shared<RuntimeState::var_desc>();
    var_desc->module = module;
    var_desc->name = deviceName;
    var_desc->address = 0;

    auto module_entries_lock = std::unique_lock(module_desc->entries_mutex);
    CHECK(module_desc->vars.insert((uintptr_t)hostVar).second);

    auto entries_lock = std::unique_lock(state.global->entries_mutex);
    CHECK(state.global->vars.insert(std::make_pair((uintptr_t)hostVar, var_desc)).second);
}


static void init_symbols() __attribute__((constructor));

static
void
init_symbols()
{
    std::string lib("libcudart.so");

    // allow env to override load path
    auto lib_env = secure_getenv("FRACTOS_SERVICE_COMPUTE_LIBCUDART");
    if (lib_env) {
        lib = lib_env;
    }

    libcudart_handle = dlopen(lib.c_str(), RTLD_LAZY);
    CHECK(libcudart_handle) << "--reason--> " << dlerror();

    char info[1024];
    CHECK(dlinfo(libcudart_handle, RTLD_DI_ORIGIN, info) == 0);
    LOG(INFO) << "opened backend cuda runtime library in "
              << info << "/" << lib.substr(lib.find_last_of("/\\") + 1);

    implemented_functions = new std::unordered_map<std::string, void*>();

    for (size_t i = 0; true; i++) {
        auto& elem = runtime_default_functions[i];
        if (not elem.name) {
            break;
        }

        auto res = implemented_functions->insert(std::make_pair(elem.name, elem.ptr));
        CHECK(res.second) << "could not insert function pointer for " << elem.name;
    }


    auto do_load_sym = [&](auto name) {
        auto ptr = dlsym(libcudart_handle, name);
        CHECK(ptr) << "--reason--> " << dlerror();

        auto res = implemented_functions->insert(std::make_pair(name, ptr));
        CHECK(not res.second) << "could not insert function pointer for " << name;

        return ptr;
    };

    // NOTE: globals override auto-generated weak symbols
#define SYM(name) ptr_ ## name = (decltype(ptr_ ## name))do_load_sym(#name);
#include "./runtime-syms.hpp"
#undef SYM
}
