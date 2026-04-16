#include <sys/mman.h>
#include <cerrno>
#include <cstring>
#include <sstream>

#include "driver-lib.hpp"


DriverLibSyms& get_driver_lib_syms() {
    static std::shared_ptr<DriverLibSyms> state;
    static std::mutex mut;

    std::unique_lock lock(mut);
    if (not state) {
        state = std::make_shared<DriverLibSyms>();
        // allow env to override load path
        auto lib = get_env("FRACTOS_SERVICE_COMPUTE_LIBCUDA", "libcuda.so");

        void* libcuda_handle = dlopen(lib.c_str(), RTLD_LAZY);
        CHECK(libcuda_handle) << "--reason--> " << dlerror();

        char info[1024];
        CHECK(dlinfo(libcuda_handle, RTLD_DI_ORIGIN, info) == 0);
        LOG(INFO) << "opened backend cuda library in "
                  << info << "/" << lib.substr(lib.find_last_of("/\\") + 1);

        auto do_load_sym = [&](auto name) {
            LOG(INFO) << "Loading " << name << "...";
            auto ptr = dlsym(libcuda_handle, name);
            LOG(INFO) << "ptr == " << ptr;
            CHECK(ptr) << "--reason--> " << dlerror();
            return ptr;
        };

#define SYM(name) state->ptr_ ## name = (decltype(state->ptr_ ## name))do_load_sym(#name);
#include "./driver-syms.hpp"
#undef SYM
    }
    return *state;
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
