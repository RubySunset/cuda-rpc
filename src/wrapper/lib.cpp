#include <cuda.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <link.h>
#include <unordered_map>


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

    load_sym_next(cuGetProcAddress_v2);
    // NOTE: globals override auto-generated weak symbols in default_functions

#undef load_sym_next
}
