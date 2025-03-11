#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include "./srv_context.hpp"
using namespace fractos;


namespace test {

class gpu_Device {
public:
    static std::shared_ptr<gpu_Device> factory(fractos::wire::endian::uint8_t id);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_make_context(auto args);
    void handle_destroy(auto args);


private:
    fractos::wire::endian::uint8_t _id;

    std::shared_ptr<gpu_Device> _self;
    bool _destroyed;
    CUdevice _device;

public:
    fractos::core::cap::request _req_make_context;
    fractos::core::cap::request _req_destroy;

    gpu_Device(fractos::wire::endian::uint8_t value);
    std::shared_ptr<test::gpu_Context> _vctx;

    ~gpu_Device();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}