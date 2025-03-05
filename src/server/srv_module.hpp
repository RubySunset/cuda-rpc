#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
#include "./srv_function.hpp"


using namespace fractos;


namespace test {

class gpu_cuda_module {
public:
    static std::shared_ptr<gpu_cuda_module> factory(std::string& name, CUcontext& ctx);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_get_function(auto args);
    void handle_destroy(auto args);
    
private:
    void module_unload();  

    std::shared_ptr<gpu_cuda_module> _self;
    bool _destroyed;
    std::string _name;
    CUcontext _ctx;
    CUmodule _module;

public:
    fractos::core::cap::request _req_get_func;
    fractos::core::cap::request _req_destroy;

    std::shared_ptr<test::gpu_cuda_function> _func; 

    gpu_cuda_module(std::string& name, CUcontext& ctx);

    ~gpu_cuda_module();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}