#include <any>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include <cstdint> // For uint8_t


#include <fractos/service/compute/cuda.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
#include "./srv_device.hpp"
using namespace fractos;

// using namespace ::service::compute::cuda;
namespace test {
    
    class gpu_device_service{
    public:
    
        static std::shared_ptr<gpu_device_service> factory();

        fractos::core::cap::request req_connect;
        fractos::core::cap::request req_generic;

        fractos::core::future<void> register_service(std::shared_ptr<fractos::core::channel> ch);
    
        void request_exit();
        bool exit_requested() const;
        bool query_event_completion(std::shared_ptr<fractos::core::channel> ch, fractos::wire::endian::uint8_t id);

        core::future<std::shared_ptr<gpu_Device>> get_or_make_device_ordinal(auto ch, int ordinal);
        std::shared_ptr<gpu_Device> get_device(CUdevice device);
    
        ~gpu_device_service();
    
    protected:
        void handle_connect(auto ch, auto args);
        void handle_generic(auto ch, auto args);

        void handle_get_driver_version(auto ch, auto args);
        void handle_init(auto ch, auto args);

        void handle_device_get(auto ch, auto args);
        void handle_device_get_count(auto ch, auto args);

        void handle_module_get_loading_mode(auto ch, auto args);

    private:
        gpu_device_service();

        std::shared_mutex _devices_mutex;
        std::unordered_map<int, std::shared_ptr<gpu_Device>> _ordinal_devices;
        std::unordered_map<CUdevice, std::shared_ptr<gpu_Device>> _devices;

        std::shared_ptr<fractos::core::channel> ch;
        std::weak_ptr<gpu_device_service> _self;
        std::atomic<bool> _requested_exit;
    
    };

    std::string to_string(const gpu_device_service& obj);

}
