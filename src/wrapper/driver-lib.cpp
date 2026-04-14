#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <dlfcn.h>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <glog/logging.h>
#include <link.h>
#include <unordered_map>

#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <sstream>


DriverLibSyms& get_driver_lib_syms() {
    static DriverLibSyms state;
    return state;
}

static void reserve_cuda_device_addr_space() __attribute__((constructor));

static
void
reserve_cuda_device_addr_space()
{
    // Attempt to reserve address range to prevent collisions with driver memory allocations
    if (mmap(reinterpret_cast<void*>(DEVICE_MAP_BASE), DEVICE_MAP_SIZE, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0) == MAP_FAILED) {
        std::stringstream ss;
        ss << "Failed to reserve address range: errno=" << errno << ", strerror=" << strerror(errno);
        throw std::runtime_error(ss.str());
    } else {
        LOG(INFO) << "Allocated address range with base " << DEVICE_MAP_BASE << " and size " << DEVICE_MAP_SIZE;
    }
}
