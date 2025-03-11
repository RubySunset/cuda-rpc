#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;


namespace test {

class gpu_Function {
public:
    static std::shared_ptr<gpu_Function> factory(std::string func_name, CUcontext& ctx, CUmodule& mod);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_call(auto args);
    void handle_func_destroy(auto args);
    
private:
    // void free(char* base);  
    std::string _name;
    std::shared_ptr<gpu_Function> _self;
    bool _destroyed;
    CUcontext _ctx;
    CUmodule _mod;
    CUfunction _func;

public:
    fractos::core::cap::request _req_call;
    fractos::core::cap::request _req_func_destroy;

    gpu_Function(std::string func_name, CUcontext& ctx, CUmodule& mod);

    ~gpu_Function();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}