#include <cuda.h>
#include <fractos/wire/error.hpp>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>
#include <tuple>


namespace impl {
    class Context;
}

namespace impl {

    class Event : protected fractos::common::service::SrvBase {
    public:
        CUevent get_remote_cuevent() const;

        CUevent cuevent;
        std::shared_ptr<Context> ctx_ptr;
        std::shared_ptr<Event> self;

        fractos::core::future<std::tuple<fractos::wire::error_type, CUresult>>
        destroy_maybe(auto ch);

        // NOTE: for internal use
    public:
        fractos::core::cap::request req_generic;

        void handle_generic(auto ch, auto args);
    protected:
        void handle_synchronize(auto ch, auto args);
        void handle_destroy(auto ch, auto args);
    };

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Event>>>
    make_event(std::shared_ptr<fractos::core::channel> ch,
               std::shared_ptr<Context> ctx, unsigned int flags);

    std::string to_string(const Event& obj);

}
