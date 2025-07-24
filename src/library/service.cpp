#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <utility>

#include <./common.hpp>
#include <service_impl.hpp>
#include <device_impl.hpp>
#include "./library_impl.hpp"


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Service;
using namespace fractos;


#define IMPL_CLASS impl::Service
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Service>;


std::shared_ptr<clt::Service>
impl::make_service(std::shared_ptr<core::channel> ch,
                   core::cap::request req_connect,
                   core::cap::request req_generic)
{
    auto state = std::make_shared<impl::ServiceState>();
    state->req_connect = std::move(req_connect);
    state->req_generic = std::move(req_generic);

    auto res = impl::Service::make(ch, state);

    {
        LOG(WARNING) << "TODO: implement server-side service creation/destruction";
        auto& pimpl = impl::Service::get(*res);
        pimpl.release_destroy();
    }

    return res;
}

core::future<void>
impl::ServiceState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    LOG(FATAL) << "not implemented";
}


core::future<std::shared_ptr<clt::Service>>
clt::make_service(std::shared_ptr<core::channel> ch,
                  core::gns::service& gns, const std::string& name,
                  const std::chrono::microseconds& wait_time)
{
    static const std::string method = "service::compute::cuda::make_service[gns]";

    LOG_OP(method)
        << " name=" << name
        << " <- {}";

    return gns.get_wait_for<core::cap::request>(ch, name, wait_time)
        .then([ch, name](auto& fut) {
            auto req = fut.get();
            LOG_OP(method)
                << " -> " << core::to_string(req);
            return make_service(ch, req)
                .then([req=std::move(req)](auto& fut) {
                    return fut.get();
                });
        })
        .unwrap();
}

core::future<std::shared_ptr<clt::Service>>
clt::make_service(std::shared_ptr<core::channel> ch,
                  const core::cap::request& connect)
{
    static const std::string method = "service::compute::cuda::make_service[connect]";
    using msg = clt::wire::Service::connect;

    LOG_OP(method)
        << " connect=" << core::to_string(connect)
        << " <- {}";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(connect)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for srv_handle
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_OP(method)
                << " -> " << wire::to_string(*args);

            if (not args->has_exactly_args()) {
                throw core::other_error(method + ": invalid response format");
            }
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return impl::make_service(
                ch,
                std::move(args->caps.connect),
                std::move(args->caps.generic));
        });
}

const core::cap::request&
clt::Service::get_connect() const
{
    auto& pimpl = impl::Service::get(*this);
    return pimpl.state->req_connect;
}

core::future<int>
clt::Service::get_driver_version()
{
    METHOD(get_driver_version);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_GET_DRIVER_VERSION)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.value.get();
        });
}

core::future<void>
clt::Service::init(unsigned int flags)
{
    METHOD(init);
    LOG_REQ(method)
        << " flags=" << flags;

    auto& pimpl = impl::Service::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_INIT)
        .set_imm(&msg::request::imms::flags, flags)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
        });
}


core::future<std::shared_ptr<clt::Device>>
clt::Service::device_get(int ordinal)
{
    METHOD(device_get);
    LOG_REQ(method)
        << " ordinal=" << ordinal;

    auto& pimpl = impl::Service::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DEVICE_GET)
        .set_imm(&msg::request::imms::ordinal, ordinal)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self, ordinal](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            if ((int)args->imms.device.get() == -1) {
                return std::shared_ptr<clt::Device>(nullptr);
            }

            return impl::make_device(
                ch,
                args->imms.device,
                std::move(args->caps.generic));
        });
}

core::future<int>
clt::Service::device_get_count()
{
    METHOD(device_get_count);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DEVICE_GET_COUNT)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return args->imms.count.get();
        });
}


core::future<CUmoduleLoadingMode>
clt::Service::module_get_loading_mode()
{
    METHOD(module_get_loading_mode);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Service::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_MODULE_GET_LOADING_MODE)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);

            return static_cast<CUmoduleLoadingMode>(args->imms.mode.get());
        });
}

core::future<std::shared_ptr<clt::Library>>
clt::Service::library_load_data(core::cap::memory& contents,
                                const std::vector<CUjit_option>& jit_options,
                                const std::vector<void*>& jit_values,
                                const std::vector<CUlibraryOption>& lib_options,
                                const std::vector<void*>& lib_values)
{
    METHOD(library_load_data);
    LOG_REQ(method)
        << " contents=" << core::to_string(contents)
        << " jit_options.size=" << jit_options.size()
        << " jit_values.size=" << jit_values.size()
        << " lib_options.size=" << lib_options.size()
        << " lib_values.size=" << lib_values.size();

    auto& pimpl = impl::Service::get(*this);
    auto self = pimpl.state->self.lock();

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    auto req = pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_LIBRARY_LOAD_DATA)
        .set_imm(&msg::request::imms::num_jit_options, jit_options.size())
        .set_imm(&msg::request::imms::num_lib_options, lib_options.size())
        .set_cap(&msg::request::caps::continuation, resp)
        .set_cap(&msg::request::caps::contents, contents);

    size_t offset = sizeof(msg::request::imms);


    if (jit_options.size() != jit_values.size()) {
        return core::make_exceptional_future<std::shared_ptr<clt::Library>>(
            CudaError(CUDA_ERROR_INVALID_VALUE));
    }

    {
        auto size = jit_options.size() * sizeof(jit_options[0]);
        req.set_imm(offset, jit_options.data(), size);
        offset += size;
    }

    size_t size_jit_values = 0;
    for (size_t i = 0; i < jit_options.size(); i++) {
        auto option = jit_options[i];
        // auto value = jit_values[i];
        switch (option) {
        default:
            LOG(FATAL) << "CUjit_option not implemented: " << option;
        }
    }


    {
        auto size = lib_options.size() * sizeof(lib_options[0]);
        req.set_imm(offset, lib_options.data(), size);
        offset += size;
    }

    // NOTE: forced empty
    size_t size_lib_values = 0;
    if (lib_values.size() > 0) {
        return core::make_exceptional_future<std::shared_ptr<clt::Library>>(
            CudaError(CUDA_ERROR_INVALID_VALUE));
    }


    return req
        .set_imm(&msg::request::imms::size_jit_values, size_jit_values)
        .set_imm(&msg::request::imms::size_lib_values, size_lib_values)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();

            return impl::make_library(
                ch,
                (CUlibrary)args->imms.culibrary.get(),
                std::move(args->caps.generic));
        });
}


std::string
clt::to_string(const clt::Service& obj)
{
    auto& pimpl = impl::Service::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Service& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const impl::ServiceState& obj)
{
    std::stringstream ss;
    ss << "cuda::Service(" << &obj << ")";
    return ss.str();
}
