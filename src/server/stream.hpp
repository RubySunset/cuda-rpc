#include <sys/stat.h>
#include <stdlib.h>
#include <queue>
#include <chrono>
#include <fractos/service/compute/cuda.hpp>


namespace impl {

    class Stream {
    public:
        static std::shared_ptr<Stream> factory(fractos::wire::endian::uint32_t flags,
                                               fractos::wire::endian::uint32_t id, CUcontext& ctx);

        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);

    protected:
        void handle_synchronize(auto args);
        void handle_destroy(auto args);
    
    private:
        void stream_synchronize();  
        void stream_destroy();  
        fractos::wire::endian::uint32_t _flags;
        fractos::wire::endian::uint32_t _id;

        std::shared_ptr<Stream> _self;
        bool _destroyed;
        CUcontext _ctx;
        CUstream _stream;

    public:
        fractos::core::cap::request _req_sync;
        fractos::core::cap::request _req_destroy;
        Stream(fractos::wire::endian::uint32_t flags, fractos::wire::endian::uint32_t id, CUcontext& ctx);

        ~Stream();
        const CUstream& getCUStream() const;

        //std::vector<std::shared_ptr<gpu_device_memory>> allocations;
    };

}
