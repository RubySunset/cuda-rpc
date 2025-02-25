#include <fractos/service/compute/cuda.hpp>
#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <cstdio>
#include <cstdlib>
#include <fractos/logging.hpp>

using namespace fractos;
using namespace fractos::service::compute::cuda;

fractos::service::compute::cuda::ErrorChecker::ErrorChecker(CUresult err, const std::string& file, int line) 
    :err(err){
    handleError(err, file, line);
}

void fractos::service::compute::cuda::ErrorChecker::handleError(CUresult err, const std::string& file, int line) {
    if (CUDA_SUCCESS != err) {
        DLOG(ERROR) << "CUDA Driver API error = " << err
                  << " from file <" << file << ">, line " << line << ".\n";
        exit(-1);
    }
    DLOG(INFO) << "CUDA Driver API SUCCESS " << " from file <" << file << ">, line " << line << ".\n";;
}
// ErrorChecker::ErrorChecker(CUresult err, const char *file, const int line) {
//     if (CUDA_SUCCESS != err) {
//         LOG(INFO) << "CUDA Driver API error = " << err
//                   << " from file <" << file << ">, line " << line << ".\n";
//         exit(-1);
//     }
// }


// service::compute::cuda::no_service_error::no_service_error(const std::string& what)
//     :std::runtime_error(what)
// {
// }
