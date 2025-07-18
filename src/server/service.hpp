#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>


namespace impl {
    class Device;
}

namespace impl {

    class Service {
    public:
        static std::shared_ptr<Service> factory();

        fractos::core::cap::request req_connect;
        fractos::core::cap::request req_generic;

        fractos::core::future<void> register_service(std::shared_ptr<fractos::core::channel> ch);

        void request_exit();
        bool exit_requested() const;
        bool query_event_completion(std::shared_ptr<fractos::core::channel> ch, fractos::wire::endian::uint8_t id);

        fractos::core::future<std::shared_ptr<Device>> get_or_make_device_ordinal(auto ch, int ordinal);
        std::shared_ptr<Device> get_device(CUdevice device);
        void erase_device(std::shared_ptr<Device> device);

        ~Service();

    protected:
        void handle_connect(auto ch, auto args);
        void handle_generic(auto ch, auto args);

        void handle_get_driver_version(auto ch, auto args);
        void handle_init(auto ch, auto args);

        void handle_device_get(auto ch, auto args);
        void handle_device_get_count(auto ch, auto args);

        void handle_module_get_loading_mode(auto ch, auto args);

    private:
        Service();

        std::shared_mutex _devices_mutex;
        std::unordered_map<int, std::shared_ptr<Device>> _ordinal_devices;
        std::unordered_map<CUdevice, std::shared_ptr<Device>> _devices;

        std::shared_ptr<fractos::core::channel> ch;
        std::weak_ptr<Service> _self;
        std::atomic<bool> _requested_exit;
    };

    std::string to_string(const Service& obj);

}
