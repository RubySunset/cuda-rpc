#include <cuda.h>
#include <fractos/wire/error.hpp>
#include <fractos/common/service/srv_base.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <memory>
#include <tuple>


namespace impl {
    class Service;
    class Library;
}

namespace impl {

    class Kernel : protected fractos::common::service::SrvBase {
    public:
        CUkernel get_remote_cukernel() const;

        CUkernel cukernel;
        std::shared_ptr<Service> service;
        std::shared_ptr<Library> library;
        std::shared_ptr<Kernel> self;

        fractos::core::future<std::tuple<fractos::wire::error_type, CUresult>>
        destroy_maybe(auto ch);

        // NOTE: for internal use
    public:
        fractos::core::cap::request req_generic;

        void handle_generic(auto ch, auto args);
    protected:
        void handle_destroy(auto ch, auto args);
    };

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Kernel>>>
    make_kernel(std::shared_ptr<fractos::core::channel> ch,
                std::shared_ptr<Service> service,
                std::shared_ptr<Library> library,
                std::string name);

    std::string to_string(const Kernel& obj);

}
