#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <utility>

#include <common.hpp>
#include <function_impl.hpp>
#include <module_impl.hpp>


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Module;
using namespace fractos;


#define IMPL_CLASS impl::Module
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Module>;


std::shared_ptr<clt::Module>
impl::make_module(std::shared_ptr<fractos::core::channel> ch,
                  CUmodule cumodule,
                  fractos::core::cap::request req_generic,
                  fractos::core::cap::request req_get_func,
                  fractos::core::cap::request req_module_unload)
{
    auto state = std::make_shared<impl::ModuleState>();
    state->req_generic = std::move(req_generic);
    state->req_get_func = std::move(req_get_func);
    state->req_module_unload = std::move(req_module_unload);
    state->cumodule = cumodule;

    return impl::Module::make(ch, state);
}

core::future<void>
impl::ModuleState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    using msg = ::service::compute::cuda::wire::Module::destroy;

    DVLOG(logging::SERVICE) << "Module::destroy <-";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_module_unload)
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


CUmodule
clt::Module::get_module() const
{
    auto& pimpl = impl::Module::get(*this);

    return pimpl.state->cumodule;
}

core::future<std::shared_ptr<clt::Function>>
clt::Module::get_function(const std::string& name)
{
    METHOD(get_function);
    LOG_REQ(method)
        << " name=" << name;

    auto& pimpl = impl::Module::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_get_func)
        .set_imm(&msg::request::imms::name_size, name.size())
        .set_imm(offsetof(msg::request::imms, name), name.c_str(), name.size())
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_CAPS_EXACT();
            CHECK_IMMS_ALL();
            CHECK_ARGS_COND(
                args->imms_size() == args->imms_expected_size() + sizeof(uint64_t) * args->imms.nargs);

            auto cufunction = (CUfunction)args->imms.cufunction.get();

            size_t args_total_size = 0;
            std::vector<size_t> args_size;
            for (size_t i = 0; i < args->imms.nargs; i++) {
                auto elem = args->imms.arg_size[i];
                args_total_size += elem;
                args_size.push_back(elem);
            }

            return impl::make_function(
                ch,
                cufunction,
                args_total_size, args_size,
                std::move(args->caps.generic));
        });
}

core::future<CUdeviceptr>
clt::Module::get_global(const std::string& name)
{
    METHOD(get_global);
    LOG_REQ(method)
        << " name=" << name;

    auto& pimpl = impl::Module::get(*this);
    auto self = pimpl.state;

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_GLOBAL)
        .set_imm(&msg::request::imms::name_size, name.size())
        .set_imm(&msg::request::imms::name, name.c_str(), name.size())
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([this, self](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.dptr;
        });
}


std::string
clt::to_string(const clt::Module& obj)
{
    auto& pimpl = impl::Module::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Module& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const impl::ModuleState& obj)
{
    std::stringstream ss;
    ss << "cuda::Module(" << (void*)obj.cumodule << ")";
    return ss.str();
}
