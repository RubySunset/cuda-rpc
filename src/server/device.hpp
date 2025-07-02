#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include "./context.hpp"
using namespace fractos;


namespace test {

class gpu_Device {
public:
    static std::shared_ptr<gpu_Device> factory(fractos::wire::endian::uint8_t id);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_generic(auto ch, auto args);
    void handle_get_attribute(auto ch, auto args);
    void handle_get_name(auto ch, auto args);
    void handle_get_uuid(auto ch, auto args);
    void handle_total_mem(auto ch, auto args);
    void handle_make_context(auto args);
    void handle_destroy(auto args);


private:
    std::shared_ptr<gpu_Device> _self;
    bool _destroyed;

public:
    const CUdevice device;
    fractos::core::cap::request req_generic;
    fractos::core::cap::request req_make_context;
    fractos::core::cap::request req_destroy;

    gpu_Device(CUdevice ordinal);
    std::shared_ptr<test::gpu_Context> _vctx;

    ~gpu_Device();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};

    std::string to_string(const gpu_Device& obj);

}
