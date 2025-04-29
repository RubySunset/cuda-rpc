#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <module_impl.hpp>
#include <function_impl.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;

inline
impl::Module&
impl::Module::get(srv::Module& obj)
{
    return *reinterpret_cast<impl::Module*>(obj._pimpl.get());
}

inline
const impl::Module&
impl::Module::get(const srv::Module& obj)
{
    return *reinterpret_cast<impl::Module*>(obj._pimpl.get());
}


// Module::Module(std::shared_ptr<void> pimpl, std::string name) : _pimpl(pimpl) {

//     DLOG(INFO) << "initialize module : " << name << " from file path";
// }


srv::Module::Module(std::shared_ptr<void> pimpl, uint64_t module_id)
    : _pimpl(pimpl)
{
    DLOG(INFO) << "initialize module id : " << module_id << " from memory buffer";
}


// Module::Module(std::shared_ptr<void> pimpl, core::cap::memory contents, std::string name) : _pimpl(pimpl) {

//     DLOG(INFO) << "initialize module : " << name << " from data buffer";
// }

srv::Module::~Module() {
    DLOG(INFO) << "Module: i am free";
    if (not _destroyed) {
        _destroyed = true;
        // TODO: check why calling ::get() sometimes gets stuck
        destroy().as_callback();
    }
}

core::future<std::shared_ptr<srv::Function>>
srv::Module::get_function(const std::string& func_name)
{
        using msg = ::service::compute::cuda::wire::Module::get_function;
    
        DVLOG(logging::SERVICE) << "Module::get_function <-";
    
        auto& pimpl = impl::Module::get(*this);
    
        auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
        return pimpl.ch->make_request_builder<msg::request>(pimpl.req_get_func)
            // .set_imm(&msg::request::imms::name, name) // unsigned int vs uint32_t
            .set_imm(&msg::request::imms::func_name_size, func_name.size())
            .set_imm(offsetof(msg::request::imms, func_name), func_name.c_str(), func_name.size())
            .set_cap(&msg::request::caps::continuation, resp)
            .on_channel()
            .invoke(resp) // wait for srv_handle
            .unwrap()
            .then([func_name](auto& fut) { // function_name
                auto [ch, args] = fut.get();

                CHECK(args->has_all_imms());
                CHECK(args->imms_size() == args->imms_expected_size() + sizeof(uint64_t) * args->imms.nargs);
                CHECK(args->has_exactly_caps());

                DVLOG(logging::SERVICE) << "Context::get_function ->"
                                        << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
                fractos::wire::error_raise_exception_maybe(args->imms.error);

                size_t args_total_size = 0;
                std::vector<size_t> args_size;
                for (size_t i = 0; i < args->imms.nargs; i++) {
                    auto elem = args->imms.arg_size[i];
                    args_total_size += elem;
                    args_size.push_back(elem);
                }

                //get Function object
                std::shared_ptr<impl::Function> pimpl_(
                    new impl::Function{{}, ch,
                            args_total_size, args_size,
                            args->imms.error,
                            std::move(args->caps.call),
                            std::move(args->caps.func_destroy)}
                    );
                pimpl_->self = pimpl_;
                auto pimpl = static_pointer_cast<void>(pimpl_);
                std::shared_ptr<Function> res(new Function{pimpl, func_name});
                return res;
            });
    }

core::future<void>
srv::Module::destroy()
{
    using msg = ::service::compute::cuda::wire::Module::destroy;

    DVLOG(logging::SERVICE) << "Module::destroy <-";

    auto& pimpl = impl::Module::get(*this);
    _destroyed = true;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.req_module_unload)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Module::destroy");
                DVLOG(logging::SERVICE) << "Module::destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Module::destroy ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}
