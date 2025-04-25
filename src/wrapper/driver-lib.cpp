#include <cstdlib>
#include <cuda.h>
#include <dlfcn.h>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <glog/logging.h>
#include <link.h>
#include <unordered_map>

#include <./driver-state.hpp>


static void *libcuda_handle;

// auto-generated function map
struct cuda_function_t {
    char const* name;
    void* ptr;
};
extern "C" [[gnu::visibility("hidden")]] cuda_function_t driver_default_functions[];

// NOTE: *cannot* be a global map, because it's constructed after init_lib() below
static std::unordered_map<std::string, void*> *implemented_functions;

// NOTE: not in cuda.h
extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** pfn, int  cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus);


#define SYM(name) decltype(&name) ptr_ ## name;
#include "./driver-syms.hpp"
#undef SYM


extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus)
{
    if (std::string(symbol) == "") {
        auto res = (ptr_cuGetProcAddress_v2)(symbol, pfn, cudaVersion, flags, symbolStatus);
        DVLOG(fractos::logging::SERVICE)
            << "cuGetProcAddress_v2"
            << " *pfn=" << *pfn
            << " symbol=\"" << symbol << "\"";
        return res;
    }

    auto it = implemented_functions->find(symbol);
    if (it != implemented_functions->end()) {
        *pfn = it->second;
        DVLOG(fractos::logging::SERVICE)
            << "cuGetProcAddress_v2"
            << " *pfn=" << *pfn
            << " symbol=\"" << symbol << "\"";
        return CUDA_SUCCESS;
    } else {
        *pfn = nullptr;
        LOG(WARNING) << "could not find function pointer for " << symbol;
        return CUDA_SUCCESS;
    }
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags)
{
    return cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, nullptr);
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
{
    // https://github.com/SveSop/nvcuda/blob/884da86135b4b0bf5f36c0eb016520c3e4d5f1d5/dlls/nvcuda/internal.c#L148
    LOG(WARNING) << "TODO: properly handle cuGetExportTable";
    auto res = (*ptr_cuGetExportTable)(ppExportTable, pExportTableId);

    if (VLOG_IS_ON(fractos::logging::SERVICE)) {
        DVLOG(fractos::logging::SERVICE)
            << "cuGetExportTable"
            << " *ppExportTable=" << *ppExportTable << " *pExportTableId="
            << fractos::service::compute::cuda::wire::to_string(*pExportTableId);
        for (auto idx = 0; (*(void***)ppExportTable)[idx]; idx++) {
            DVLOG(fractos::logging::SERVICE)
                << "    [" << idx << "]=" << std::hex << (*(void***)ppExportTable)[idx];
        }
    }

    return res;
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
        auto& elem = driver_default_functions[i];
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

    // NOTE: globals override auto-generated weak symbols
#define SYM(name) ptr_ ## name = (decltype(ptr_ ## name))do_load_sym(#name);
#include "./driver-syms.hpp"
#undef SYM
}
