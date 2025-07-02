#include <chrono>
#include <cuda.h>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>

#include "./memory.hpp"
#include "./module.hpp"
#include "./stream.hpp"
#include "./event.hpp"


using namespace fractos;


namespace test {
    
class gpu_Context {
public:
    static std::shared_ptr<gpu_Context> factory(fractos::wire::endian::uint32_t id, CUdevice device);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_generic(auto ch, auto args);
    void handle_get_api_version(auto ch, auto args);
    void handle_get_limit(auto ch, auto args);
    void handle_mem_alloc(auto ch, auto args);

    void handle_stream(auto args);
    void handle_event(auto args);
    void handle_module_file(auto args);
    void handle_module_data(auto args);
    void handle_synchronize(auto args);
    void handle_destroy(auto args);


private:
    void context_synchronize(); // type?
    void context_destroy(CUcontext& context); // type?

    fractos::wire::endian::uint32_t _id;

    std::shared_ptr<gpu_Context> _self;
    bool _destroyed;
public:
    CUcontext _ctx; 

    fractos::core::cap::request _req_generic;
    fractos::core::cap::request _req_stream;
    fractos::core::cap::request _req_event;
    fractos::core::cap::request _req_module_data;
    // fractos::core::cap::request _req_module_file;
    fractos::core::cap::request _req_synchronize;
    fractos::core::cap::request _req_destroy;

    gpu_Context(fractos::wire::endian::uint32_t value, CUdevice device);
    std::shared_ptr<test::gpu_Stream> _stream; 
    std::shared_ptr<test::gpu_Event> _event; 
    std::shared_ptr<test::gpu_Module> _mod; 

    ~gpu_Context();

    const std::unordered_map<int, std::shared_ptr<gpu_Stream>>& getVStreamMap() const;

private:
    std::unordered_map<int, std::shared_ptr<gpu_Stream>> _vstream_map;

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};

    std::string to_string(const gpu_Context& obj);
}
