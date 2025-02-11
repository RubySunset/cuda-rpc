#include <fractos/service/compute/cuda.hpp>
#include <cstdio>
#include <cstdlib>
#include <fractos/logging.hpp>

using namespace fractos::service::compute::cuda;

ErrorChecker::ErrorChecker(CUresult err, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        LOG(INFO) << "CUDA Driver API error = " << err
                  << " from file <" << file << ">, line " << line << ".\n";
        exit(-1);
    }
}
