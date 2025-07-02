#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
using namespace fractos;


namespace test {

class gpu_Memory {
public:
    static std::shared_ptr<gpu_Memory> factory(fractos::wire::endian::uint32_t size, CUcontext& ctx);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_destroy(auto args);
    
private:
    void memory_free(char* base);  
    fractos::wire::endian::uint32_t _size;

    std::shared_ptr<gpu_Memory> _self;
    bool _destroyed;
    CUcontext _ctx;

public:
    fractos::core::cap::request _req_destroy;

    fractos::core::cap::memory _memory;
    std::shared_ptr<fractos::core::memory_region> _mr;
    
    CUdeviceptr base;

    gpu_Memory(fractos::wire::endian::uint32_t size, CUcontext& ctx);

    ~gpu_Memory();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}
