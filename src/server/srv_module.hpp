#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include "./srv_function.hpp"
// #include "./srv_context.hpp"


using namespace fractos;


namespace test {
class gpu_Context;
class gpu_Module {
public:
    static std::shared_ptr<gpu_Module> factory(std::string& name, CUcontext& ctx);
    static std::shared_ptr<gpu_Module> factory(uint64_t module_id, CUcontext& ctx, std::shared_ptr<char>& buffer, size_t size
                                , std::weak_ptr<test::gpu_Context> vctx);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_generic(auto ch, auto args);
    void handle_get_global(auto ch, auto args);

    void handle_get_function(auto args);
    void handle_destroy(auto args);
    
private:
    void module_unload();  

    std::shared_ptr<gpu_Module> _self;
    bool _destroyed;
    std::string _name;
    uint64_t _id;
    CUcontext _ctx;
    CUmodule _module;
    std::weak_ptr<test::gpu_Context> _vctx;
    std::shared_ptr<const char> _data;

public:
    fractos::core::cap::request _req_generic;
    fractos::core::cap::request _req_get_func;
    fractos::core::cap::request _req_destroy;

    std::shared_ptr<test::gpu_Function> _func; 

    gpu_Module(std::string& name, CUcontext& ctx);
    gpu_Module(uint64_t module_id, CUcontext& ctx,  std::shared_ptr<char>& buffer, size_t size, std::weak_ptr<test::gpu_Context> vctx);

    ~gpu_Module();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};

    std::string to_string(const gpu_Module& obj);
}
