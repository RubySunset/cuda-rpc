#include <chrono>
#include <cuda.h>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <queue>
#include <stdlib.h>
#include <sys/stat.h>


namespace impl {
    class Context;
    class Module;
    class Kernel;
}

namespace impl {

    class Function : protected fractos::common::service::SrvBase {
    public:
        CUfunction get_remote_cufunction() const;

        CUfunction cufunction;
        size_t args_total_size;
        std::vector<size_t> args_size;
        std::shared_ptr<Context> ctx;
        std::shared_ptr<Function> self;

        fractos::core::future<std::tuple<fractos::wire::error_type, CUresult>>
        destroy_maybe(auto ch);

        // NOTE: for internal use
    public:
        fractos::core::cap::request req_generic;
        void handle_generic(auto ch, auto args);
    protected:
        void handle_set_attribute(auto ch, auto args);
        void handle_launch(auto ch, auto args);
        void handle_destroy(auto ch, auto args);
    };

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Function>>>
    make_function(std::shared_ptr<fractos::core::channel> ch,
                  std::shared_ptr<Context> ctx, std::shared_ptr<Module> module, const std::string name);

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Function>>>
    make_function(std::shared_ptr<fractos::core::channel> ch,
                  std::shared_ptr<Context> ctx, std::shared_ptr<Kernel> kernel);

    std::string to_string(const Function& obj);

}
