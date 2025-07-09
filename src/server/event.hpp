#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>


namespace impl {

    class Event {
    public:
        CUevent get_remote_cuevent() const;

        static std::shared_ptr<Event> factory(fractos::wire::endian::uint32_t flags,
                                                  CUcontext& ctx);

        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

    protected:
        // void handle_synchronize(auto args);
        void handle_destroy(auto args);
    
    private:
        // void stream_synchronize();  
        void event_destroy();  
        fractos::wire::endian::uint32_t _flags;
        // fractos::wire::endian::uint32_t _id;

        std::shared_ptr<Event> _self;
        bool _destroyed;
        CUcontext _ctx;
        CUevent _event;

    public:
        // fractos::core::cap::request _req_sync;
        fractos::core::cap::request _req_destroy;
        Event(fractos::wire::endian::uint32_t flags, CUcontext& ctx);

        ~Event();
        // const CUstream& getCUEvent() const;

        //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
    };

}
