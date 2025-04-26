#pragma once

extern "C" void** CUDARTAPI __cudaRegisterFatBinary(void *fatCubin);
extern "C" void CUDARTAPI __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
extern "C" void CUDARTAPI __cudaUnregisterFatBinary(void **fatCubinHandle);
extern "C" void CUDARTAPI __cudaRegisterFunction(
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
    );
extern "C" void CUDARTAPI __cudaRegisterVar(
        void **fatCubinHandle,
        char  *hostVar,
        char  *deviceAddress,
  const char  *deviceName,
        int    ext,
        size_t size,
        int    constant,
        int    global
);
extern "C" cudaError_t CUDARTAPI __cudaPushCallConfiguration(
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem,
    struct CUstream_st *stream
    );
extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(
    dim3         *gridDim,
    dim3         *blockDim,
    size_t       *sharedMem,
    void         *stream
    );

#define SYM(name) extern decltype(&name) ptr_ ## name;
#include "./runtime-syms.hpp"
#undef SYM
