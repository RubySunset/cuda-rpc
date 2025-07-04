#pragma once

#include <cuda.h>
#include <string>

// TODO: delete
#define checkCudaErrors(err)  ErrorChecker(err, __FILE__, __LINE__)

// TODO: delete
struct ErrorChecker {
    ErrorChecker(CUresult err);
    // ErrorChecker(CUresult err,  const char *file, const int line);

    ErrorChecker(CUresult err, const std::string& file, int line);

private:
    void handleError(CUresult err, const std::string& file, int line);
    const CUresult err;
};

static inline const char* get_CUresult_name(CUresult err) __attribute__((const));

static inline const char *
get_CUresult_name(CUresult err)
{
    const char* res;
    CHECK(cuGetErrorName(err, &res) == CUDA_SUCCESS);
    return res;
}
