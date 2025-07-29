#include <cuda.h>

#include "./image.hpp"
#include "./driver-state.hpp"


namespace clt = fractos::service::compute::cuda;


// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__LIBRARY.html


extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuKernelGetFunction(CUfunction* pFunc, CUkernel kernel)
{
    auto& state = get_driver_state();

    auto ctx = state.get_current_context();
    CHECK(ctx);

    auto kernel_desc = state.get_kernel(kernel);
    if (not kernel_desc) {
        return CUDA_ERROR_NOT_FOUND;
    }

    auto func_desc = std::make_shared<DriverState::func_desc>();
    try {
        func_desc->function = kernel_desc->kernel->get_function(*ctx)
            .get();
    } catch (const clt::CudaError& e) {
        return e.cuerror;
    }

    state.insert_function(func_desc);

    *pFunc = func_desc->function->get_function();

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuLibraryLoadData(CUlibrary* library, const void* code,
                  CUjit_option* jitOptions, void** jitOptionsValues, unsigned int numJitOptions,
                  CUlibraryOption* libraryOptions, void** libraryOptionValues, unsigned int  numLibraryOptions) 
{
    auto& state = get_driver_state();

    // parse descriptor

    auto code_size = get_image_size(code);

    // load library into service

    auto& ch = get_channel();

    // RO cap, synchronously prefetch default MR to avoid unexpected perm errors
    auto& mr = ch.get_default_memory_region();
    mr.prefetch(fractos::core::memory_region::prefetch_type::ODP_RD_SYNC,
                code, code_size);
    auto code_cap = ch.make_memory(code, code_size, mr).get();
    CHECK(not code_cap.has_any_perms(fractos::core::cap::PERM_WR));

    std::vector<CUjit_option> jit_options(jitOptions, jitOptions + numJitOptions);
    std::vector<void*> jit_values(jitOptionsValues, jitOptionsValues + numJitOptions);
    std::vector<CUlibraryOption> lib_options(libraryOptions, libraryOptions + numLibraryOptions);
    std::vector<void*> lib_values;

    std::shared_ptr<clt::Library> library_ptr;
    try {
        library_ptr = state.service->library_load_data(code_cap,
                                                       jit_options, jit_values,
                                                       lib_options, lib_values).get();
    } catch (const clt::CudaError& e) {
        return e.cuerror;
    }

    auto library_desc = std::make_shared<DriverState::library_desc>();
    library_desc->library = library_ptr;
    state.insert_library(library_desc);

    *library = library_ptr->get_library();

    return CUDA_SUCCESS;
}

extern "C" [[gnu::visibility("default")]]
CUresult CUDAAPI
cuLibraryGetKernel(CUkernel* pKernel, CUlibrary library, const char* name)
{
    auto& state = get_driver_state();

    auto library_desc = state.get_library(library);
    if (not library_desc) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    std::string name_s(name);

    auto kernel_desc = std::make_shared<DriverState::kernel_desc>();
    try {
        kernel_desc->kernel = library_desc->library->get_kernel(name)
            .get();
    } catch (const clt::CudaError& e) {
        return e.cuerror;
    }

    state.insert_kernel(kernel_desc);

    *pKernel = kernel_desc->kernel->get_kernel();

    return CUDA_SUCCESS;
}
