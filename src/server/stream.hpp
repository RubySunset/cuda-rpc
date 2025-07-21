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

        const CUstream custream;
        std::shared_ptr<Context> ctx_ptr;
        std::shared_ptr<Stream> self;

        fractos::core::future<std::tuple<fractos::wire::error_type, CUresult>>
        destroy_maybe(auto ch);

    protected:
        void handle_generic(auto ch, auto args);
        void handle_synchronize(auto ch, auto args);
        void handle_destroy(auto ch, auto args);

        // NOTE: for internal use
    public:
        fractos::core::cap::request req_generic;

        Stream(std::shared_ptr<Context> ctx, CUstream stream);
        ~Stream();
        fractos::core::future<void> register_methods(std::shared_ptr<fractos::core::channel> ch);
    };

    std::pair<CUresult, std::shared_ptr<Stream>> make_stream(std::shared_ptr<Context> ctx, unsigned int flags);

    std::string to_string(const Stream& obj);

}
