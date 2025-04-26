#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <string>
#include <unordered_map>

#include "./runtime-state.hpp"
#include "./runtime-syms-extern.hpp"

#define SYM(name) decltype(&name) ptr_ ## name;
#include "./runtime-syms.hpp"
#undef SYM


static void *libcudart_handle;

// auto-generated function map
struct cuda_function_t {
    char const* name;
    void* ptr;
};
extern "C" [[gnu::visibility("hidden")]] cuda_function_t runtime_default_functions[];

// NOTE: *cannot* be a global map, because it's constructed after init_lib() below
static std::unordered_map<std::string, void*> *implemented_functions;


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
