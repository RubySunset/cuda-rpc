#include <cuda.h>

#include "./image.hpp"
#include "./driver-state.hpp"
#include "./driver-syms-extern.hpp"

namespace srv = fractos::service::compute::cuda;


// * module management
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuModuleGetFunction(CUfunction* hfunc, CUmodule hmod, const char *name)
{
    auto& state = get_driver_state();

    auto mod_desc = state.get_module(hmod);
    if (not mod_desc) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::string func_name(name);
    {
        auto funcs_lock = std::unique_lock(mod_desc->functions_mutex);
        auto it = mod_desc->functions.find(func_name);
        if (it != mod_desc->functions.end()) {
            *hfunc = it->second;
            return CUDA_SUCCESS;
        }
    }

    auto func_desc = std::make_shared<DriverState::func_desc>();
    func_desc->function = mod_desc->module->get_function(name).get();

    CUfunction func = (CUfunction)func_desc.get();

    {
        auto funcs_lock = std::unique_lock(mod_desc->functions_mutex);
        auto res = mod_desc->functions.insert(std::make_pair(func_name, func));
        CHECK(res.second);
    }

    {
        auto funcs_lock = std::unique_lock(state.functions_mutex);
        auto res = state.functions.insert(std::make_pair(func, func_desc));
        CHECK(res.second);
    }

    *hfunc = func;
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuModuleGetGlobal_v2(CUdeviceptr* dptr, size_t* bytes, CUmodule hmod, const char *name)
{
    auto& state = get_driver_state();

    CHECK(bytes == nullptr) << "not implemented";

    auto mod_desc = state.get_module(hmod);
    if (not mod_desc) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::string name_str(name);
    *dptr = mod_desc->module->get_global(name).get();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuModuleGetLoadingMode(CUmoduleLoadingMode* mode)
{
    auto& state = get_driver_state();
    *mode = state.service->module_get_loading_mode().get();
    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuModuleLoadData(CUmodule *module, const void *image)
{
    auto& state = get_driver_state();

    CUcontext ctx;
    auto err = cuCtxGetCurrent(&ctx);
    if (err != CUDA_SUCCESS) {
        return err;
    }
    CHECK(ctx != 0);

    auto ctx_ptr = state.get_context(ctx);
    CHECK(ctx_ptr);

    // parse descriptor

    auto image_size = get_image_size(image);

    // load module into service

    auto& ch = get_channel();

    // RO cap, synchronously prefetch default MR to avoid unexpected perm errors
    auto& mr = ch.get_default_memory_region();
    mr.prefetch(fractos::core::memory_region::prefetch_type::ODP_RD_SYNC,
                image, image_size);
    auto image_cap = ch.make_memory(image, image_size, mr).get();
    CHECK(not image_cap.has_any_perms(fractos::core::cap::PERM_WR));

    auto module_ptr = ctx_ptr->module_load_data(image_cap).get();
    auto module_id = module_ptr->get_module();
    auto module_desc = std::make_shared<DriverState::module_desc>();
    module_desc->image = image;
    module_desc->image_size = image_size;
    module_desc->module = module_ptr;
    {
        auto modules_lock = std::unique_lock(state.modules_mutex);
        auto res = state.modules.insert(std::make_pair(module_id, module_desc));
        CHECK(res.second);
    }

    *module = module_id;

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuModuleUnload(CUmodule hmod)
{
    auto& state = get_driver_state();

    std::shared_ptr<DriverState::module_desc> mod_desc;

    {
        auto modules_lock = std::unique_lock(state.modules_mutex);

        auto it = state.modules.find(hmod);
        if (it == state.modules.end()) {
            return CUDA_ERROR_INVALID_IMAGE;
        }

        mod_desc = it->second;

        state.modules.erase(it);
    }

    {
        auto functions_lock = std::unique_lock(state.functions_mutex);
        auto module_functions_lock = std::unique_lock(mod_desc->functions_mutex);
        for (auto& func: mod_desc->functions) {
            auto it = state.functions.find(func.second);
            CHECK(it != state.functions.end());
            it->second->function->destroy().get();
            state.functions.erase(it);
        }
    }

    mod_desc->module->destroy().get();

    return CUDA_SUCCESS;
}
