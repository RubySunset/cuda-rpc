#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
#include "./srv_context.hpp"
using namespace fractos;


namespace test {

class gpu_cuda_device {
public:
    static std::shared_ptr<gpu_cuda_device> factory(fractos::wire::endian::uint8_t id);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_make_cuda_context(auto args);
    void handle_destroy(auto args);


private:
    fractos::wire::endian::uint8_t _id;

    std::shared_ptr<gpu_cuda_device> _self;
    bool _destroyed;
    CUdevice _device;

public:
    fractos::core::cap::request _req_make_cuda_context;
    fractos::core::cap::request _req_destroy;

    gpu_cuda_device(fractos::wire::endian::uint8_t value);
    std::shared_ptr<test::gpu_cuda_context> _vctx;

    ~gpu_cuda_device();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}