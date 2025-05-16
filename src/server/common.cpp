#include <fractos/logging.hpp>
#include <glog/logging.h>

#include "common.hpp"


using namespace fractos;


ErrorChecker::ErrorChecker(CUresult err, const std::string &file, int line)
    :err(err)
{
    handleError(err, file, line);
}

void
ErrorChecker::handleError(CUresult err, const std::string& file, int line)
{
    if (CUDA_SUCCESS != err) {
        DLOG(ERROR) << "CUDA Driver API error = " << err
                  << " from file <" << file << ">, line " << line << ".\n";
        exit(-1);
    }
    DVLOG(logging::SERVICE) << "CUDA Driver API SUCCESS " << " from file <" << file << ">, line " << line << ".\n";;
}
