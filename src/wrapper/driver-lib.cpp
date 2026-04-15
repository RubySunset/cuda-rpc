#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <sstream>

#include "driver-lib.hpp"


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
