#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>
using namespace fractos;


namespace test {

class gpu_cuda_context {
public:
    static std::shared_ptr<gpu_cuda_context> factory(fractos::wire::endian::uint8_t id);

    fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

protected:
    void handle_destroy(auto args);


private:
    fractos::wire::endian::uint8_t _id;

    std::shared_ptr<gpu_cuda_context> _self;
    bool _destroyed;

public:
    fractos::core::cap::request _req_destroy;

    gpu_cuda_context(fractos::wire::endian::uint8_t value);

    ~gpu_cuda_context();

    //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
};
}