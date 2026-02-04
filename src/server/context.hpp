#include <chrono>
#include <cuda.h>
#include <fractos/wire/error.hpp>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>


using namespace fractos;


namespace impl {
    class Device;
    class Stream;
    class Event;
    class CublasHandle;
    class Module;
    class Memory;
}

namespace impl {

    class Context : public fractos::common::service::SrvBase {
    public:
        CUcontext get_remote_cucontext() const;

        std::shared_ptr<Stream> get_stream(CUstream stream);
        void insert_stream(std::shared_ptr<Stream> stream);
        void erase_stream(std::shared_ptr<Stream> stream);

        void insert_event(std::shared_ptr<Event> event);
        void erase_event(std::shared_ptr<Event> event);

        std::shared_ptr<CublasHandle> get_cublas_handle(cublasHandle_t handle);
        void insert_cublas_handle(std::shared_ptr<CublasHandle> cublas_handle);
        void erase_cublas_handle(std::shared_ptr<CublasHandle> cublas_handle);

        CUcontext cucontext;
        std::shared_ptr<Device> device;
        std::shared_ptr<Context> self;

    private:
        std::mutex _stream_map_mutex;
        std::unordered_map<CUstream, std::shared_ptr<Stream>> _stream_map;

        std::mutex _event_map_mutex;
        std::unordered_map<CUevent, std::shared_ptr<Event>> _event_map;

        std::mutex _cublas_map_mutex;
        std::unordered_map<cublasHandle_t, std::shared_ptr<CublasHandle>> _cublas_map;

        // NOTE: for internal use
    public:
        fractos::core::cap::request _req_generic;
        void handle_generic(auto ch, auto args);
    protected:
        void handle_get_api_version(auto ch, auto args);
        void handle_get_limit(auto ch, auto args);
        void handle_module_load_data(auto ch, auto args);
        void handle_memcpy_async(auto ch, auto args);
        void handle_mem_alloc(auto ch, auto args);
        void handle_mem_get_info(auto ch, auto args);
        void handle_memset(auto ch, auto args);
        void handle_stream_create(auto ch, auto args);
        void handle_event_create(auto ch, auto args);
        void handle_cublas_create(auto ch, auto args);
        void handle_synchronize(auto ch, auto args);
        void handle_destroy(auto ch, auto args);
    };

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Context>>>
    make_context(std::shared_ptr<fractos::core::channel> ch,
                 std::shared_ptr<Device> device,
                 unsigned int flags);

    std::string to_string(const Context& obj);
}
