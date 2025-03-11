#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include "./srv_function.hpp"


using namespace fractos;


namespace test {

class gpu_Module {
public:
    static std::shared_ptr<gpu_Module> factory(std::string& name, CUcontext& ctx);
    static std::shared_ptr<gpu_Module> factory(std::string& name, CUcontext& ctx, char* buffer, size_t size);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_get_function(auto args);
    void handle_destroy(auto args);
    
private:
    void module_unload();  

    std::shared_ptr<gpu_Module> _self;
    bool _destroyed;
    std::string _name;
    CUcontext _ctx;
    CUmodule _module;

public:
    fractos::core::cap::request _req_get_func;
    fractos::core::cap::request _req_destroy;

    std::shared_ptr<test::gpu_Function> _func; 

    gpu_Module(std::string& name, CUcontext& ctx);
    gpu_Module(std::string& name, CUcontext& ctx,  char* buffer, size_t size);

    ~gpu_Module();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}