#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
// #include <./srv_stream.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;


namespace test {
class gpu_Context;
class gpu_Stream;
class gpu_Function {
public:
    static std::shared_ptr<gpu_Function> factory(std::string func_name, CUcontext& ctx, CUmodule& mod
        , std::weak_ptr<test::gpu_Context> vctx);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_generic(auto ch, auto args);
    void handle_launch(auto ch, auto args);
    void handle_destroy(auto ch, auto args);

private:
    // void free(char* base);  
    std::string _name;
    std::shared_ptr<gpu_Function> _self;
    bool _destroyed;
    CUcontext _ctx;
    CUmodule _mod;
    CUfunction _func;
    size_t _args_total_size;
public:
    std::vector<size_t> _args_size;
private:

    std::weak_ptr<test::gpu_Context> _vctx; 
    std::weak_ptr<test::gpu_Stream> _vstream;

public:
    fractos::core::cap::request _req_generic;

    gpu_Function(std::string func_name, CUcontext& ctx, CUmodule& mod, std::weak_ptr<test::gpu_Context> vctx);

    ~gpu_Function();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};

    std::string to_string(const gpu_Function& obj);

}
