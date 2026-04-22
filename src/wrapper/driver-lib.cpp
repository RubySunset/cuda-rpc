#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <sstream>

#include "driver-lib.hpp"


DriverLibSyms&
get_driver_lib_syms()
{
    static std::shared_ptr<DriverLibSyms> state;
    static std::mutex mut;

    std::unique_lock lock(mut);
    if (not state) {
        state = std::make_shared<DriverLibSyms>();
        // allow env to override load path
        auto lib = get_env("FRACTOS_SERVICE_COMPUTE_LIBCUDA", "libcuda.so");

        void* libcuda_handle = dlopen(lib.c_str(), RTLD_LAZY);
        CHECK(libcuda_handle) << "--reason--> " << dlerror();

        char info[1024];
        CHECK(dlinfo(libcuda_handle, RTLD_DI_ORIGIN, info) == 0);
        LOG(INFO) << "opened backend cuda library in "
                  << info << "/" << lib.substr(lib.find_last_of("/\\") + 1);

        auto do_load_sym = [&](auto name) {
            LOG(INFO) << "Loading " << name << "...";
            auto ptr = dlsym(libcuda_handle, name);
            LOG(INFO) << "ptr == " << ptr;
            CHECK(ptr) << "--reason--> " << dlerror();
            return ptr;
        };

#define SYM(name) state->ptr_ ## name = (decltype(state->ptr_ ## name))do_load_sym(#name);
#include "./driver-syms.hpp"
#undef SYM
    }
    return *state;
}

// auto-generated function map
struct cuda_function_t {
    char const* name;
    void* ptr;
};
extern "C" [[gnu::visibility("hidden")]] cuda_function_t driver_default_functions[];

std::shared_ptr<std::unordered_map<std::string, void*>>
get_implemented_functions()
{
    get_driver_lib_syms();

    static std::shared_ptr<std::unordered_map<std::string, void*>> implemented_functions;
    static std::mutex mut;

    std::unique_lock lock(mut);
    if (not implemented_functions) {
        implemented_functions = std::make_shared<std::unordered_map<std::string, void*>>();

        for (size_t i = 0; true; i++) {
            auto& elem = driver_default_functions[i];
            if (not elem.name) {
                break;
            }

            auto res = implemented_functions->insert(std::make_pair(elem.name, elem.ptr));
            CHECK(res.second) << "could not insert function pointer for " << elem.name;
        }

#define SYM(name) \
    { \
        auto res = implemented_functions->insert(std::make_pair(#name, (void*)get_driver_lib_syms().ptr_ ## name)); \
        CHECK(not res.second) << "could not insert function pointer for " << #name; \
    }
#include "./driver-syms.hpp"
#undef SYM
    }
    return implemented_functions;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                    cuuint64_t flags, CUdriverProcAddressQueryResult* symbolStatus)
{
    if (std::string(symbol) == "") {
        auto res = (*get_driver_lib_syms().ptr_cuGetProcAddress_v2)(symbol, pfn, cudaVersion, flags, symbolStatus);
        DVLOG(fractos::logging::SERVICE)
            << "cuGetProcAddress_v2"
            << " *pfn=" << *pfn
            << " symbol=\"" << symbol << "\"";
        return res;
    }

    auto it = get_implemented_functions()->find(symbol);
    if (it != get_implemented_functions()->end()) {
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

// https://github.com/SveSop/nvcuda/blob/884da86135b4b0bf5f36c0eb016520c3e4d5f1d5/dlls/nvcuda/internal.c#L148
// https://git.tedomum.net/jae/ZLUDA/-/blob/af0216b1a000cb2f8b705611a9198c916cca3146/zluda_dark_api/src/lib.rs#L197

struct table_info {
    const CUuuid uuid;
    const size_t table_size;
    const void **table;
    std::atomic_flag inited;
    std::mutex inited_mutex;
};

#define DEFINE_TABLE(name, ...)                                         \
    static table_info name = {                                          \
        .uuid = {__VA_ARGS__},                                          \
        .table_size = sizeof(_table_ ##name) / sizeof(_table_ ##name [0]), \
        .table = _table_ ##name,                                        \
        .inited = false                                                 \
    }


static const void *_table_cudart_api[] = {
    0x0,
    (void *)0xcafe0001,
    (void *)&cuDevicePrimaryCtxRetain,
    (void *)0xcafe0003,
    (void *)0xcafe0004,
    (void *)0xcafe0005,
    (void *)0xcafe0006,
    (void *)0xcafe0007,
    (void *)0xcafe0008,
    (void *)0xcafe0009,
    0x0,
};

DEFINE_TABLE(cudart_api,
             (char)0x6b, (char)0xd5, (char)0xfb, (char)0x6c,
             (char)0x5b, (char)0xf4, (char)0xe7, (char)0x4a,
             (char)0x89, (char)0x87, (char)0xd9, (char)0x39,
             (char)0x12, (char)0xfd, (char)0x9d, (char)0xf9);


static const void*
_table_context_local_storage_api[] = {
    (void*)0xcafe0101,
    (void*)0xcafe0102,
    (void*)0xcafe0103,
    (void*)0xcafe0104,
};

DEFINE_TABLE(context_local_storage_api,
             (char)0xc6, (char)0x93, (char)0x33, (char)0x6e,
             (char)0x11, (char)0x21, (char)0xdf, (char)0x11,
             (char)0xa8, (char)0xc3, (char)0x68, (char)0xf3,
             (char)0x55, (char)0xd8, (char)0x95, (char)0x93);


static const void*
_table_tools_runtime_hooks_api[] = {
    0x0,
    (void*)0xcafe0201,
    0x0,
    (void*)0xcafe0203,
    (void*)0xcafe0204,
    (void*)0xcafe0205,
    (void*)0xcafe0206,
    0x0,
};

DEFINE_TABLE(tools_runtime_hooks_api,
             (char)0xa0, (char)0x94, (char)0x79, (char)0x8c,
             (char)0x2e, (char)0x74, (char)0x2e, (char)0x74,
             (char)0x93, (char)0xf2, (char)0x08, (char)0x00,
             (char)0x20, (char)0x0c, (char)0x0a, (char)0x66);


static const void*
_table_tools_tls_api[] = {
    0x0,
    (void*)0xcafe0301,
    0x0,
    0x0,
};

DEFINE_TABLE(tools_tls_api,
             (char)0x42, (char)0xd8, (char)0x5a, (char)0x81,
             (char)0x23, (char)0xf6, (char)0xcb, (char)0x47,
             (char)0x82, (char)0x98, (char)0xf6, (char)0xe7,
             (char)0x8a, (char)0x3a, (char)0xec, (char)0xdc);


static const void*
_table_x1_api[] = {
    0x0,
    (void*)0xcafe0401,
    (void*)0xcafe0402,
    (void*)0xcafe0403,
    (void*)0xcafe0404,
    (void*)0xcafe0405,
    (void*)0xcafe0406,
    (void*)0xcafe0407,
    (void*)0xcafe0408,
    (void*)0xcafe0409,
    (void*)0xcafe0410,
    (void*)0xcafe0411,
    (void*)0xcafe0412,
    (void*)0xcafe0413,
    (void*)0xcafe0414,
    (void*)0xcafe0415,
    (void*)0xcafe0416,
    (void*)0xcafe0417,
    (void*)0xcafe0418,
    (void*)0xcafe0419,
    (void*)0xcafe0420,
    (void*)0xcafe0421,
    (void*)0xcafe0422,
    (void*)0xcafe0423,
    (void*)0xcafe0424,
    (void*)0xcafe0425,
    (void*)0xcafe0426,
    (void*)0xcafe0427,
    (void*)0xcafe0428,
    (void*)0xcafe0429,
    (void*)0xcafe0430,
    (void*)0xcafe0431,
    (void*)0xcafe0432,
    (void*)0xcafe0433,
    (void*)0xcafe0434,
    (void*)0xcafe0435,
    (void*)0xcafe0436,
    (void*)0xcafe0437,
    (void*)0xcafe0438,
    0x0, // !!! 0x7ffff02477d0
    (void*)0xcafe0440,
    (void*)0xcafe0441,
    (void*)0xcafe0442,
    (void*)0xcafe0443,
    (void*)0xcafe0444,
    (void*)0xcafe0445,
    (void*)0xcafe0446,
    (void*)0xcafe0447,
    (void*)0xcafe0448,
    (void*)0xcafe0449,
    (void*)0xcafe0450,
    0x0,
    (void*)0xcafe0452,
    (void*)0xcafe0453,
    (void*)0xcafe0454,
    (void*)0xcafe0455,
    (void*)0xcafe0456,
    (void*)0xcafe0457,
    (void*)0xcafe0458,
    (void*)0xcafe0459,
    (void*)0xcafe0460,
    (void*)0xcafe0461,
    (void*)0xcafe0462,
    (void*)0xcafe0463,
    (void*)0xcafe0464,
    (void*)0xcafe0465,
    (void*)0xcafe0466,
    (void*)0xcafe0467,
    (void*)0xcafe0468,
    (void*)0xcafe0469,
    (void*)0xcafe0470,
    (void*)0xcafe0471,
    (void*)0xcafe0472,
    (void*)0xcafe0473,
    (void*)0xcafe0474,
    (void*)0xcafe0475,
    (void*)0xcafe0476,
    (void*)0xcafe0477,
    (void*)0xcafe0478,
    (void*)0xcafe0479,
    (void*)0xcafe0480,
    (void*)0xcafe0481,
    (void*)0xcafe0482,
    (void*)0xcafe0483,
    (void*)0xcafe0484,
    (void*)0xcafe0485,
    (void*)0xcafe0486,
    (void*)0xcafe0487,
    (void*)0xcafe0488,
    (void*)0xcafe0489,
    (void*)0xcafe0490,
    0x0,
};

DEFINE_TABLE(x1_api,
             (char)0x21, (char)0x31, (char)0x8c, (char)0x60,
             (char)0x97, (char)0x14, (char)0x32, (char)0x48,
             (char)0x8c, (char)0xa6, (char)0x41, (char)0xff,
             (char)0x73, (char)0x24, (char)0xc8, (char)0xf2);



extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
{
    LOG(WARNING) << "TODO: this is only partially handling cuGetExportTable, and functionality is not verified";

    auto retrieve_info = [](auto uuid) -> const table_info* {
        for (auto info : {&cudart_api, &context_local_storage_api,
                          &tools_runtime_hooks_api, &tools_tls_api, &x1_api}) {
            if (memcmp(&info->uuid, uuid, sizeof(CUuuid)) == 0) {
                if (not info->inited.test()) {
                    // merge tables
                    std::unique_lock<std::mutex> lock(info->inited_mutex);
                    if (not info->inited.test_and_set()) {
                        void** orig_table;
                        auto res = (*get_driver_lib_syms().ptr_cuGetExportTable)((const void**)&orig_table, uuid);
                        if (res != CUDA_SUCCESS) {
                            return nullptr;
                        }

                        for (size_t idx = 0; idx < info->table_size; idx++) {
                          if (not info->table[idx]) {
                              info->table[idx] = orig_table[idx];
                          }
                        }
                    }
                }

                return info;
            }
        }

        return nullptr;
    };


    auto res = CUDA_SUCCESS;
    auto info = retrieve_info(pExportTableId);
    if (not info) {
        res = (*get_driver_lib_syms().ptr_cuGetExportTable)(ppExportTable, pExportTableId);
    } else {
        *ppExportTable = info->table;
    }

    // debug-print selected table + entry
    if (VLOG_IS_ON(fractos::logging::SERVICE)) {
        DVLOG(fractos::logging::SERVICE)
            << "cuGetExportTable"
            << " *ppExportTable=" << *ppExportTable << " *pExportTableId="
            << fractos::service::compute::cuda::wire::to_string(*pExportTableId)
            << " -> res=" << res;
        for (size_t idx = 0; idx < info->table_size; idx++) {
            DVLOG(fractos::logging::SERVICE)
                << "    [" << idx << "]=" << (void*)info->table[idx];
        }
    }

    // table-specific actions
    if (info == &tools_tls_api) {
        // NOTE: index [2] needs preexisting cuInit() to work
        (*get_driver_lib_syms().ptr_cuInit)(0);
    }

    return res;
}


static void reserve_cuda_device_addr_space() __attribute__((constructor));

static void reserve_cuda_device_addr_space() __attribute__((constructor));

static
void
reserve_cuda_device_addr_space()
{
    // Attempt to reserve address range to prevent collisions with driver memory allocations
    if (mmap(reinterpret_cast<void*>(DEVICE_MAP_BASE), DEVICE_MAP_SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0) == MAP_FAILED) {
        std::stringstream ss;
        ss << "Failed to reserve address range: errno=" << errno << ", strerror=" << strerror(errno);
        throw std::runtime_error(ss.str());
    } else {
        LOG(INFO) << "Allocated address range with base " << DEVICE_MAP_BASE << " and size " << DEVICE_MAP_SIZE;
    }
}
