# New GPU Service

# Test for example
1. move the `app-compute-cuda/` into `experiment/deps/app-compute-cuda`.
2. move the `app-compute-cuda-src/` into `experiment/src/app-compute-cuda`.
3. move the `app-compute-cuda-src/service-compute-cuda.mak` into `experiment/`.
4. update experiment/Makefile with `include src/service-compute-cuda.mak` and `include src/app-compute-cuda/rules.mak`.
5. make build/service-compute-cuda.
6. make run/app-compute-cuda.


## Currently Supported CUDA API Functions

- `cuInit`
- `cuDeviceGet`
- `cuModuleLoad`
- `cuMemAlloc`
- `cuMemFree`
- `cuModuleGetFunction`
- `cuLaunchKernel`
- `cuCtxCreate`
- `cuCtxSynchronize`
- `cuCtxDestroy`

## To Be Confirmed (TBC)

- `cuMemsetD8 `
- `cuMemGetInfo`
- `cuStreamCreate`
- `cuStreamSynchronize`
- `cuStreamDestroy`
- `cuModuleLoadData`
- `cuEvent...`
