#include <fractos/common/service/clt_impl.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <utility>

#include <common.hpp>
#include <function_impl.hpp>
#include <stream_impl.hpp>


namespace clt = fractos::service::compute::cuda;
namespace srv_wire = fractos::service::compute::cuda::wire;
namespace srv_wire_msg = srv_wire::Function;
using namespace fractos;


#define IMPL_CLASS impl::Function
#include <fractos/common/service/clt_base.inc.hpp>
#undef IMPL_CLASS
template class fractos::common::service::CltBase<clt::Function>;


std::shared_ptr<clt::Function>
impl::make_function(std::shared_ptr<fractos::core::channel> ch,
                    size_t args_total_size,
                    std::vector<size_t> args_size,
                    fractos::core::cap::request req_generic)
{
    auto state = std::make_shared<impl::FunctionState>();
    state->args_total_size = args_total_size;
    state->args_size = std::move(args_size);
    state->req_generic = std::move(req_generic);

    return impl::Function::make(ch, state);
}

core::future<void>
impl::FunctionState::do_destroy(std::shared_ptr<core::channel>& ch)
{
    METHOD(destroy);
    LOG_REQ(method)
        << " {}";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_DESTROY)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}


core::future<void>
clt::Function::set_attribute(CUfunction_attribute attrib, int value)
{
    METHOD(set_attribute);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Function::get(*this);
    auto self = pimpl.state->self.lock();
    CHECK(self);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    return pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_SET_ATTRIBUTE)
        .set_imm(&msg::request::imms::attrib, (uint64_t)attrib)
        .set_imm(&msg::request::imms::value, (uint64_t)value)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response_ptr(self)
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

core::future<void>
clt::Function::launch(const void** args, dim3 gridDim, dim3 blockDim,
                      size_t sharedMemBytes,
                      std::optional<std::reference_wrapper<Stream>> stream)
{
    METHOD(launch);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Function::get(*this);
    auto self = pimpl.state->self.lock();

    CUstream custream = 0;
    if (stream) {
        custream = stream->get().get_stream();
    }

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    auto req = pimpl.ch->make_request_builder<msg::request>(pimpl.state->req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire_msg::OP_LAUNCH)
        .set_imm(&msg::request::imms::grid_x, (uint64_t)gridDim.x)
        .set_imm(&msg::request::imms::grid_y, (uint64_t)gridDim.y)
        .set_imm(&msg::request::imms::grid_z, (uint64_t)gridDim.z)
        .set_imm(&msg::request::imms::block_x, (uint64_t)blockDim.x)
        .set_imm(&msg::request::imms::block_y, (uint64_t)blockDim.y)
        .set_imm(&msg::request::imms::block_z, (uint64_t)blockDim.z)
        .set_imm(&msg::request::imms::shared_mem, sharedMemBytes)
        .set_imm(&msg::request::imms::custream, (uint64_t)custream)
        .set_cap(&msg::request::caps::continuation, resp);

    size_t offset = offsetof(msg::request::imms, kernel_args);
    for (size_t i = 0; i < pimpl.state->args_size.size(); i++) {
        auto size = pimpl.state->args_size[i];
        req.set_imm(offset, args[i], size);
        offset += size;
    }

    return req
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_check_cuda_response()
        .then([self](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

void
clt::Function::_launch_check_args(const std::vector<size_t>& args_size)
{
    auto& pimpl = impl::Function::get(*this);
    if (pimpl.state->args_size.size() != args_size.size()) {
        throw std::runtime_error("invalid number of arguments");
    }
    for (size_t i = 0; i < args_size.size(); i++) {
        if (pimpl.state->args_size[i] != args_size[i]) {
            std::stringstream ss;
            ss << "invalid size for argument " << i;
            throw std::runtime_error(ss.str());
        }
    }
}


std::string
clt::to_string(const clt::Function& obj)
{
    auto& pimpl = impl::Function::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Function& obj)
{
    return impl::to_string(*obj.state);
}

std::string
impl::to_string(const impl::FunctionState& obj)
{
    std::stringstream ss;
    ss << "cuda::Function(" << &obj << ")";
    return ss.str();
}
