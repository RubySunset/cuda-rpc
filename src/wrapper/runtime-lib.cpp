#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
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

    struct fat_cubin {
        uint32_t magic; // Always 0x466243b1
        uint32_t seq;   // Sequence number of the cubin
        uint64_t ptr;   // The pointer to the real cubin
        uint64_t data_ptr;    // Some pointer related to the data segment
    };

    fat_cubin* desc = (fat_cubin*)fatCubin;
    CHECK(desc->magic == 0x466243b1);

    DVLOG(logging::APP)
        << "__cudaRegisterFatBinary ->"
        << " fatCubin=0x" << std::hex << fatCubin
        << " desc->ptr=0x" << std::hex << desc->ptr;

    state.last_error = (cudaError_t)cuModuleLoadData(&module, (const void*)desc->ptr);
    if (state.last_error != cudaSuccess) {
        LOG(FATAL) << "failed cuModuleLoadData(..., " << (const void*)desc->ptr << "): "
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
