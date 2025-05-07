#include <fractos/service/compute/cuda.hpp>
#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <cstdio>
#include <cstdlib>
#include <fractos/logging.hpp>


using namespace fractos;
namespace srv = fractos::service::compute::cuda;


srv::ErrorChecker::ErrorChecker(CUresult err, const std::string& file, int line) 
    :err(err){
    handleError(err, file, line);
}

void srv::ErrorChecker::handleError(CUresult err, const std::string& file, int line) {
    if (CUDA_SUCCESS != err) {
        DLOG(ERROR) << "CUDA Driver API error = " << err
                  << " from file <" << file << ">, line " << line << ".\n";
        exit(-1);
    }
    DVLOG(logging::SERVICE) << "CUDA Driver API SUCCESS " << " from file <" << file << ">, line " << line << ".\n";;
}

static
const char *
get_cuda_error(CUresult error)
{
    const char* msg = nullptr;
    auto err = cuGetErrorName(error, &msg);
    CHECK(err == CUDA_SUCCESS) << err;
    return msg;
}

srv::CudaError::CudaError(CUresult cuerror)
    :std::runtime_error(get_cuda_error(cuerror))
    ,cuerror(cuerror)
{
}
