#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include "./srv_memory.hpp"
#include "./srv_module.hpp"
using namespace fractos;


namespace test {

class gpu_Context {
public:
    static std::shared_ptr<gpu_Context> factory(fractos::wire::endian::uint32_t id, CUdevice& device);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_memory(auto args);
    void handle_module_file(auto args);
    void handle_synchronize(auto args);
    void handle_destroy(auto args);


private:
    char* allocate_memory(size_t size, CUcontext& context); // type?
    void context_synchronize(); // type?
    void context_destroy(CUcontext& context); // type?

    fractos::wire::endian::uint32_t _id;

    std::shared_ptr<gpu_Context> _self;
    bool _destroyed;
    CUcontext _ctx; 
    

public:
    fractos::core::cap::request _req_memory;
    fractos::core::cap::request _req_module_file;
    fractos::core::cap::request _req_synchronize;
    fractos::core::cap::request _req_destroy;

    gpu_Context(fractos::wire::endian::uint32_t value, CUdevice& device);
    std::shared_ptr<test::gpu_Memory> _dev_mem;
    std::shared_ptr<test::gpu_Module> _mod; 

    ~gpu_Context();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}