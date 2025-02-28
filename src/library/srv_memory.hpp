#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
using namespace fractos;


namespace test {

class gpu_cuda_memory {
public:
    static std::shared_ptr<gpu_cuda_memory> factory(fractos::wire::endian::uint32_t size);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_destroy(auto args);


private:
    fractos::wire::endian::uint32_t _size;

    std::shared_ptr<gpu_cuda_memory> _self;
    bool _destroyed;

public:
    fractos::core::cap::request _req_destroy;

    // fractos::core::cap::memory _memory;
    // std::shared_ptr<fractos::core::memory_region> _mr;
    
    char* base;

    gpu_cuda_memory(fractos::wire::endian::uint32_t size);

    ~gpu_cuda_memory();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}