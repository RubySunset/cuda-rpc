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

    class Function : public fractos::common::service::SrvBase {
    public:
        Function(std::weak_ptr<Context> ctx_ptr, CUfunction func,
                 std::vector<size_t> args_size, size_t args_total_size);
        ~Function();

        fractos::core::future<void>
        register_methods(std::shared_ptr<fractos::core::channel> ch);

        const CUfunction func;
        const size_t args_total_size;
        const std::vector<size_t> args_size;
        std::weak_ptr<Context> ctx_ptr;

        // TODO: this is a memory leak; use weak_ptr and track functions in module
        std::shared_ptr<Function> self;
        fractos::core::cap::request req_generic;

    protected:
        void handle_generic(auto ch, auto args);
        void handle_launch(auto ch, auto args);
        void handle_destroy(auto ch, auto args);
    };

    std::pair<CUresult, std::shared_ptr<Function>>
    make_function(std::shared_ptr<Context> ctx_ptr, CUmodule mod, const std::string name);

    std::string to_string(const Function& obj);

}
