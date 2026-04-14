#pragma once

#include <cstdlib>
#include <cuda.h>
#include <dlfcn.h>
#include <fractos/common/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <glog/logging.h>
#include <link.h>
#include <unordered_map>

#include <common.hpp>
#include <./driver-state.hpp>

// Define region of memory that is reserved for device memory allocations
// TODO currently the device memory reservation size is hardcoded to whether
// or not RXE is used, because it's assumed that RXE must be used for the
// lower-end GPUs, but a better system would decouple these
#ifdef USE_RXE
    constexpr size_t DEVICE_MAP_SIZE = 0x40000000 * 40l; // 40GB
    constexpr size_t DEVICE_MAP_BASE = 0x7fffa0000000 - DEVICE_MAP_SIZE;
#else
    constexpr size_t DEVICE_MAP_SIZE = 0x40000000 * 100l; // 100GB
    constexpr size_t DEVICE_MAP_BASE = 0x7fffb0000000 - DEVICE_MAP_SIZE;
#endif

struct cuda_function_t {
    char const* name;
    void* ptr;
};

struct DriverLibSyms {
#define SYM(name) decltype(&name) ptr_ ## name;
#include "./driver-syms.hpp"
#undef SYM
};

DriverLibSyms& get_driver_lib_syms();
