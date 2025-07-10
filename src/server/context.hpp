#include <chrono>
#include <cuda.h>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>


using namespace fractos;


namespace impl {
    class Stream;
    class Event;
    class Module;
    class Memory;
}

namespace impl {

    class Context {
    public:
        std::shared_ptr<Stream> get_stream(CUstream stream);
        void insert_stream(std::shared_ptr<Stream> stream);
        void erase_stream(std::shared_ptr<Stream> stream);

    public:
        static std::shared_ptr<Context> factory(fractos::wire::endian::uint32_t id, CUdevice device);

        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

    protected:
        void handle_generic(auto ch, auto args);
        void handle_get_api_version(auto ch, auto args);
        void handle_get_limit(auto ch, auto args);
        void handle_module_load_data(auto ch, auto args);
        void handle_mem_alloc(auto ch, auto args);
        void handle_stream_create(auto ch, auto args);
        void handle_event_create(auto ch, auto args);

        void handle_synchronize(auto args);
        void handle_destroy(auto args);


    private:
        void context_synchronize(); // type?
        void context_destroy(CUcontext& context); // type?

        fractos::wire::endian::uint32_t _id;

    public:
        std::shared_ptr<Context> _self;
    private:
        bool _destroyed;
    public:
        CUcontext _ctx; 

        fractos::core::cap::request _req_generic;
        fractos::core::cap::request _req_synchronize;
        fractos::core::cap::request _req_destroy;

        Context(fractos::wire::endian::uint32_t value, CUdevice device);
        std::shared_ptr<Stream> _stream; 
        std::shared_ptr<Event> _event; 
        std::shared_ptr<Module> _mod; 

        ~Context();

    private:
        std::mutex _stream_map_mutex;
        std::unordered_map<CUstream, std::shared_ptr<Stream>> _stream_map;
    };

    std::string to_string(const Context& obj);
}
