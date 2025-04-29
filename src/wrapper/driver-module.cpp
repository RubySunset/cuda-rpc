#include <cuda.h>

#include <./driver-state.hpp>
#include <./driver-syms-extern.hpp>

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
        return CUDA_ERROR_INVALID_IMAGE;
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
    get_args_from_image(*func_desc, func_name, *mod_desc);

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
    struct nv_fatbin_header {
        uint8_t magic[8];
        uint64_t size;
    };

    static constexpr uint8_t nv_fatbin_magic[8] = {0x50, 0xed, 0x55, 0xba, 0x01, 0x00, 0x10, 0x00};


    auto& state = get_driver_state();

    // parse descriptor

    auto header = (nv_fatbin_header*)image;
    CHECK(memcmp(header->magic, nv_fatbin_magic, 8) == 0);

    CUcontext ctx;
    auto err = cuCtxGetCurrent(&ctx);
    if (err != CUDA_SUCCESS) {
        return err;
    }
    CHECK(ctx != 0);

    auto ctx_ptr = state.get_context(ctx);
    CHECK(ctx_ptr);

    auto image_size = sizeof(nv_fatbin_header) + header->size;

    // load module into service

    auto image_cap_base = get_channel().make_memory(image, image_size).get();
    auto image_cap = get_channel().diminish(image_cap_base, 0, image_size, fractos::core::cap::PERM_WR).get();

    LOG(ERROR) << "TODO: no use for module_id arg";
    auto module_ptr = ctx_ptr->make_module_data(image_cap, 0).get();
    auto module_id = (CUmodule)module_ptr.get();
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
            it->second->function->func_destroy().get();
            state.functions.erase(it);
        }
    }

    mod_desc->module->destroy().get();

    return CUDA_SUCCESS;
}
