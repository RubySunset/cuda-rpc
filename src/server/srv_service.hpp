


#include <any>
#include <memory>
#include <string>
#include <unordered_map>

#include <cstdint> // For uint8_t


#include <fractos/service/compute/cuda.hpp>
#include "./srv_device.hpp"
using namespace fractos;

// using namespace ::service::compute::cuda;
namespace test {
    
    class gpu_device_service{
    public:
    
        static std::shared_ptr<gpu_device_service> factory();
    
        fractos::core::cap::request req_make_cuda_device;
        fractos::core::cap::request req_get_cuda_device;
    
        fractos::core::future<void> register_service(std::shared_ptr<fractos::core::channel> ch);
    
        void request_exit();
        bool exit_requested() const;
        bool query_event_completion(std::shared_ptr<fractos::core::channel> ch, fractos::wire::endian::uint8_t id);
    
        ~gpu_device_service();
    
    protected:
        void handle_make_cuda_device(auto args);
        void handle_get_cuda_device(auto args);
    
    public:
        std::shared_ptr<void> _pimpl;
    
    private:
        gpu_device_service();
        std::shared_ptr<test::gpu_cuda_device> _vdev;
    
        std::shared_ptr<fractos::core::channel> ch;
        std::weak_ptr<gpu_device_service> _self;
        std::atomic<bool> _requested_exit;
    
    };
    
    }
    