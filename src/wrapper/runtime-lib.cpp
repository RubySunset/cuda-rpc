#include <runtime-lib.hpp>

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

RuntimeLibSyms& get_runtime_lib_syms() {
    static RuntimeLibSyms syms;
    return syms;
}


extern "C" [[gnu::visibility("default")]]
void** CUDARTAPI
__cudaRegisterFatBinary(void* fatCubin)
{
    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::SERVICE)
            << "__cudaRegisterFatBinary <->"
            << " fatCubin=0x" << std::hex << fatCubin
            << " err=" << cudaGetErrorName(err);
        return (void**)0;
    }

    CHECK(state_ptr);
    auto& state [[maybe_unused]] = *state_ptr;

    auto desc = (const __fatBinC_Wrapper_t*)fatCubin;
    CHECK(desc->magic == FATBINC_MAGIC);
    CHECK(desc->version == FATBINC_VERSION);

    DVLOG(logging::SERVICE)
        << "__cudaRegisterFatBinary <->"
        << " fatCubin=0x" << std::hex << fatCubin
        << " desc->data=0x" << std::hex << desc->data;

    auto module_desc = std::make_shared<RuntimeState::module_desc>();
    module_desc->module = 0;
    {
        auto modules_lock = std::unique_lock(state.global->modules_mutex);
        auto res = state.global->fat_cubin_handles.insert(std::make_pair((fatCubinHandle_t)desc, module_desc));
        CHECK(res.second);
    }

    return (void**)desc;
}

extern "C" [[gnu::visibility("default")]]
void CUDARTAPI
__cudaRegisterFatBinaryEnd(void** fatCubinHandle)
{
    DVLOG(logging::SERVICE)
        << "__cudaRegisterFatBinaryEnd <->"
        << " fatCubinHandle=0x" << std::hex << fatCubinHandle;

    if (not fatCubinHandle) {
        return;
    }
}

extern "C" [[gnu::visibility("default")]]
void CUDARTAPI
__cudaUnregisterFatBinary(void** fatCubinHandle)
{
    if (not fatCubinHandle) {
        return;
    }

    DVLOG(logging::SERVICE)
        << "__cudaUnregisterFatBinary ->"
        << " fatCubinHandle=0x" << std::hex << fatCubinHandle;

    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::SERVICE)
            << "__cudaUnregisterFatBinary <- "
            << " err=" << cudaGetErrorName((cudaError_t)err);
        return;
    }
    auto& state = *state_ptr;

    auto modules_lock = std::unique_lock(state.global->modules_mutex);
    auto entries_lock = std::unique_lock(state.global->entries_mutex);

    std::shared_ptr<RuntimeState::module_desc> module_desc;
    {
        auto it = state.global->fat_cubin_handles.find((fatCubinHandle_t)fatCubinHandle);
        CHECK(it != state.global->fat_cubin_handles.end());
        module_desc = it->second;

        auto module = module_desc->module.load();
        if (module != 0) {
            state.last_error = (cudaError_t)cuModuleUnload(module);
            if (state.last_error != cudaSuccess) {
                goto out;
            }
        }
    }

    {
        auto module_desc_it = state.global->modules.find(module_desc->module);
        CHECK(module_desc_it != state.global->modules.end());
        auto module_desc = module_desc_it->second;

        auto module_entries_lock = std::unique_lock(module_desc->entries_mutex);
        for (auto& func : module_desc->funcs) {
            CHECK(state.global->funcs.erase(func) == 1);
        }
        for (auto& func : module_desc->vars) {
            CHECK(state.global->vars.erase(func) == 1);
        }

        CHECK(state.global->fat_cubin_handles.erase(fatCubinHandle) == 1);
        CHECK(state.global->modules.erase(module_desc->module) == 1);
    }

out:
    DVLOG(logging::SERVICE)
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
    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::SERVICE)
            << "__cudaRegisterFunction <-> "
            << " handle=" << fatCubinHandle
            << " hostFun=" << (void*)hostFun
            << " deviceName=" << deviceName
            << " err=" << cudaGetErrorName((cudaError_t)err);
        return;
    } else {
        DVLOG(logging::SERVICE)
            << "__cudaRegisterFunction <->"
            << " handle=" << fatCubinHandle
            << " hostFun=" << (void*)hostFun
            << " deviceName=" << deviceName;
    }
    auto& state = *state_ptr;

    auto modules_lock = std::unique_lock(state.global->modules_mutex);
    auto module_desc_it = state.global->fat_cubin_handles.find(fatCubinHandle);
    CHECK(module_desc_it != state.global->fat_cubin_handles.end());
    auto module_desc = module_desc_it->second;

    auto module_entries_lock = std::unique_lock(module_desc->entries_mutex);
    auto module_entries_res = module_desc->funcs.insert((uintptr_t)hostFun);
    CHECK(module_entries_res.second);

    auto entries_lock = std::unique_lock(state.global->entries_mutex);
    auto entries_it = state.global->funcs.find((uintptr_t)hostFun);
    if (entries_it == state.global->funcs.end()) {
        auto func_desc = std::make_shared<RuntimeState::func_desc>();
        CHECK(func_desc->fat_cubin_handles.insert(fatCubinHandle).second);
        func_desc->name = deviceName;
        func_desc->func = 0;
        auto entries_res = state.global->funcs.insert(std::make_pair((uintptr_t)hostFun, func_desc));
        CHECK(entries_res.second);
    } else {
        auto entry_lock = std::unique_lock(entries_it->second->mutex);
        CHECK(entries_it->second->name == deviceName);
        CHECK(entries_it->second->fat_cubin_handles.insert(fatCubinHandle).second);
    }
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
    auto [err, state_ptr] = get_runtime_state_with_error();
    if (err) {
        DVLOG(logging::SERVICE)
            << "__cudaRegisterVar <->"
            << " handle=" << fatCubinHandle
            << " hostVar=" << (void*)hostVar
            << " deviceName=" << deviceName
            << " err=" << cudaGetErrorName((cudaError_t)err);
        return;
    } else {
        DVLOG(logging::SERVICE)
            << "__cudaRegisterVar <->"
            << " handle=" << fatCubinHandle
            << " hostVar=" << (void*)hostVar
            << " deviceName=" << deviceName;
    }
    auto& state = *state_ptr;

    auto modules_lock = std::unique_lock(state.global->modules_mutex);
    auto module_desc_it = state.global->fat_cubin_handles.find(fatCubinHandle);
    CHECK(module_desc_it != state.global->fat_cubin_handles.end());
    auto module_desc = module_desc_it->second;

    auto var_desc = std::make_shared<RuntimeState::var_desc>();
    var_desc->fat_cubin_handle = fatCubinHandle;
    var_desc->module = 0;
    var_desc->name = deviceName;
    var_desc->address = 0;

    auto module_entries_lock = std::unique_lock(module_desc->entries_mutex);
    CHECK(module_desc->vars.insert((uintptr_t)hostVar).second);

    auto entries_lock = std::unique_lock(state.global->entries_mutex);
    // CHECK(state.global->vars.insert(std::make_pair((uintptr_t)hostVar, var_desc)).second);
    state.global->vars.insert(std::make_pair((uintptr_t)hostVar, var_desc));
}

extern "C" [[gnu::visibility("default")]]
cudaError_t CUDARTAPI
cudaGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId)
{
    auto& state = get_runtime_state();

    auto error = (cudaError_t)cuGetExportTable(ppExportTable, pExportTableId);
    return_error(error);
}


static void init_symbols() __attribute__((constructor));

static
void
init_symbols()
{
    // allow env to override load path
    auto lib = get_env("FRACTOS_SERVICE_COMPUTE_LIBCUDART", "libcudart.so");

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

#define SYM(name) get_runtime_lib_syms().ptr_ ## name = (decltype(get_runtime_lib_syms().ptr_ ## name))do_load_sym(#name);
#include "./runtime-syms.hpp"
#undef SYM
}
