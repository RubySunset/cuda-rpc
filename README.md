This repository provides access remote CUDA devices in FractOS.

[[_TOC_]]

# Overview

This repository contains three main parts:
* A service to access remote CUDA-enabled devices (`src/service`)
* An asynchronous RPC library to access the service (`src/include` and `src/library`)
* A wrapper library to redirect CUDA driver and CUDA runtime operations to the remote service (`src/wrapper`)

The service provides a remote counterpart of the CUDA driver API (https://docs.nvidia.com/cuda/cuda-driver-api/index.html).

# Using the CUDA service

Applications can access the CUDA service via the API in `fractos/service/compute/cuda.hpp` (located in `src/include/fractos/service/compute/cuda.hpp`).

Each new connection to the service via `cuda::make_service(core::gns::service)` will create an isolated service instance.
Applications using the same service via `cuda::make_service(core::cap::request)` will share a virtual address space for CUDA allocations.

# Using the CUDA wrapper library

The wrapper library uses library interposition to intercept calls to the CUDA driver and CUDA runtime API, and reimplements them in terms of the CUDA service API in this repository.

To use the wrapper library, you should link your application against the dynamic version of the CUDA driver and CUDA runtime libraries, and then start your application to point to the wrapper libraries provided by this repository.

## Compiling applications to support the wrapper library

By default, high-level CUDA libraries such as cuBLAS to cuDNN statically link against the CUDA libraries, making library interposition complex without knowledge on the internals of the CUDA libraries.

Instead, you should compile applications against the static version of such high-level libraries, and the dynamic version of the CUDA API libraries. For example, instead of compiling a cuBLAS `example.cu` as usual:

    nvcc -o example exmaple.cu -lcublas

you should instead compile it as:

    nvcc -o example exmaple.cu -lcublas_static -lcublasLt_static -lcudart -lculibos -lcuda

The reason why interosition is so difficult without these changes, is that various CUDA libraries use functions such as `cuGetExportTable` to directly call each other, bypassing the public library symbols that we can interpose.


## Starting applications with the wrapper library

To start your application with the CUDA wrapper library, you must set the `LD_LIBRARY_PATH` environment variable to the installation directory `lib/libfractos-service-compute-cuda-wrapper` (created during `make install` on this repository).
This directory has links with the name of the core CUDA libraries to the wrapper library installed in `lib/libfractos-service-compute-cuda-wrapper.so`.

Alternatively, you can directly link against the wrapper library that is installed in `lib/libfractos-service-compute-cuda-wrapper.so`:

    nvcc -o example exmaple.cu -lcublas_static -lcublasLt_static -lfractos-service-compute-cuda-wrapper


## Configuring the wrapper library

You can configure the wrapper library through additional environment variables. You can find them by running:

    rgrep get_env src/wrapper/

All environment variables are optional, unless otherwise noted. The most important ones are:

* `FRACTOS_SERVICE_COMPUT_CUDA_CONTROLLER`: overrides the value passed to `fractos::core::parse_controller_config()` when calling `fractos::core::make_process()`

* `FRACTOS_SERVICE_COMPUT_CUDA_PROCESS`: overrides the value passed to `fractos::core::parse_process_config()` when calling `fractos::core::make_process()`

* `FRACTOS_SERVICE_COMPUTE_CUDA_CHANNEL`: overrides the value passed to `fractos::core::parse_channel_config()` when calling `fractos::core::process::make_channel()`

* `FRACTOS_SERVICE_COMPUTE_LIBCUDA`: overrides the location of the system-wide CUDA driver library

* `FRACTOS_SERVICE_COMPUTE_LIBCUDART`: overrides the location of the system-wide CUDA runtime library

* `FRACTOS_SERVICE_COMPUTE_CUDA_NAME`: overrides the name used to publish the CUDA service with the FractOS GNS


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
