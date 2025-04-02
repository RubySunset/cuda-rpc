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
- `cuModuleLoadData`
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
- `cuEvent...`

## Roadmap feature

- cublas
- cuda graph support
- global service & client service.


## TroubleShooting

### CUDA Driver Error 803
_Error_: System has unsupported display driver / CUDA driver combination. Container CUDA Driver version is different from HOST. IF host update the driver version.

_Solution_ : Update the driver version by rebuilding the container image. & `modprobe nvidia-peermem` on HOST.