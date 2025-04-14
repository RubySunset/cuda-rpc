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
- `cuStreamCreate`
- `cuStreamSynchronize`
- `cuStreamDestroy`

## To Be Confirmed (TBC)

- `cuMemsetD8 `
- `cuMemGetInfo`
- `cuEventCreate`
- `cuEventDestroy`
- `cuEventRecord`
- `cudaEventSynchronize`

## Roadmap feature

- cublas
- cuda graph support
- global service & client service.


## TroubleShooting

### CUDA Driver Error 803
_Error_: System has unsupported display driver / CUDA driver combination. Container CUDA Driver version is different from HOST. IF host update the driver version.

_Solution_ : Update the driver version by rebuilding the container image. & `modprobe nvidia-peermem` on HOST.

### ibv_reg_mr failed with "cannot allocate memory" or "bad address"

_Error_: Trying to allocate more than ~256MB GPU memory on a device using vctx->make_memory() fails. This is because infiniband tries to pin the memory region, but the GPU does not have enough BAR memory space to do so (see https://docs.nvidia.com/cuda/gpudirect-rdma/#pci-bar-sizes).

_Solution_: Increase the BAR memory space of the GPU by changing the display mode. See https://developer.nvidia.com/displaymodeselector to get the tool. You should change the mode to 'Physical Display Ports Disabled with 64GB BAR1'. Notes:

- Ensure the following nvidia kernel modules are removed before running the tool: `rmmod nvidia_peermem nvidia_uvm nvidia_drm nvidia_modeset nvidia`.

- If you get an error `nvidia_uvm is in use`, stop the `nvidia-dcgm` service by running `sudo systemctl stop nvidia-dcgm`.

- After running the tool, reboot the system with `sudo reboot` to apply the changes.