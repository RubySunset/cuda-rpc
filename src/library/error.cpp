#include <fractos/service/compute/cuda.hpp>
#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <cstdio>
#include <cstdlib>
#include <fractos/logging.hpp>


namespace clt = fractos::service::compute::cuda;
using namespace fractos;


static
const char *
get_cuda_error(CUresult error)
{
    const char* msg = nullptr;
    auto err = cuGetErrorName(error, &msg);
    CHECK(err == CUDA_SUCCESS) << err;
    return msg;
}

clt::CudaError::CudaError(CUresult cuerror)
    :std::runtime_error(get_cuda_error(cuerror))
    ,cuerror(cuerror)
{
}

static
const char *
get_cublas_error(cublasStatus_t status)
{
    return cublasGetStatusName(status);
}

clt::CublasError::CublasError(cublasStatus_t cublas_error)
    :std::runtime_error(get_cublas_error(cublas_error))
    ,cublas_error(cublas_error)
{
}
