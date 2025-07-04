#include <chrono>
#include <cuda.h>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>


namespace impl {
    class Context;
}

namespace impl {

    class Stream : public fractos::common::service::SrvBase {
    public:
        CUstream get_remote_custream() const;

        const CUstream stream;
        std::weak_ptr<Context> ctx_ptr;
        std::weak_ptr<Stream> self;

    protected:
        void handle_synchronize(auto args);
        void handle_destroy(auto args);

        // NOTE: for internal use
    public:
        fractos::core::cap::request _req_sync;
        fractos::core::cap::request _req_destroy;

        Stream(Context& ctx, CUstream stream);
        ~Stream();
        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);
    };

    std::pair<CUresult, std::shared_ptr<Stream>> make_stream(Context& ctx, unsigned int flags);

    std::string to_string(const Stream& obj);

}
