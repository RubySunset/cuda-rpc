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

    class Library : protected fractos::common::service::SrvBase {
    public:
        CUlibrary get_remote_culibrary() const;

        CUlibrary culibrary;
        std::shared_ptr<Library> self;

        fractos::core::future<std::tuple<fractos::wire::error_type, CUresult>>
        destroy_maybe(auto ch);

        // NOTE: for internal use
    public:
        std::shared_ptr<const char[]> contents;
        fractos::core::cap::request req_generic;

        void handle_generic(auto ch, auto args);
    protected:
        void handle_destroy(auto ch, auto args);
    };

    fractos::core::future<std::tuple<fractos::wire::error_type, CUresult, std::shared_ptr<Library>>>
    make_library(std::shared_ptr<fractos::core::channel> ch,
                 std::shared_ptr<char[]>& contents,
                 const std::vector<CUjit_option>& jit_options, const std::vector<void*>& jit_values,
                 const std::vector<CUlibraryOption>& lib_options, const std::vector<void*>& lib_values);

    std::string to_string(const Library& obj);

}
