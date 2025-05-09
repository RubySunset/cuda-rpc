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


using namespace fractos;
namespace srv = fractos::service::compute::cuda;


std::string
srv::to_string(const srv::Function& obj)
{
    auto& pimpl = impl::Function::get(obj);
    return impl::to_string(pimpl);
}

std::string
impl::to_string(const impl::Function& obj)
{
    std::stringstream ss;
    ss << "cuda::Function(" << &obj << ")";
    return ss.str();
}

impl::Function::Function(std::shared_ptr<fractos::core::channel> ch,
                         size_t args_total_size, std::vector<size_t> args_size,
                         fractos::core::cap::request req_generic)
    :ch(ch)
    ,args_total_size(args_total_size)
    ,args_size(args_size)
    ,req_generic(std::move(req_generic))
{
}

srv::Function::Function(std::shared_ptr<void> pimpl, std::string func_name)
    :_pimpl(pimpl)
{
}

srv::Function::~Function()
{
    auto& pimpl = impl::Function::get(*this);
    pimpl.destroy_maybe()
        // keep pimpl alive
        .then([pimpl=this->_pimpl](auto& fut) {
            (void)fut.get();
        })
        .as_callback();
}

core::future<void>
srv::Function::launch(const void** args, dim3 gridDim, dim3 blockDim,
                      size_t sharedMemBytes,
                      std::optional<std::reference_wrapper<Stream>> stream)
{
    METHOD(Function, launch);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Function::get(*this);
    auto self = pimpl.self.lock();

    uint32_t stream_id = 0;
    if (stream) {
        auto& stream_pimpl = impl::Stream::get(stream->get());
        stream_id = stream_pimpl.id;
    }

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    auto req = pimpl.ch->make_request_builder<msg::request>(pimpl.req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire::OP_LAUNCH)
        .set_imm(&msg::request::imms::grid_x, (uint64_t)gridDim.x)
        .set_imm(&msg::request::imms::grid_y, (uint64_t)gridDim.y)
        .set_imm(&msg::request::imms::grid_z, (uint64_t)gridDim.z)
        .set_imm(&msg::request::imms::block_x, (uint64_t)blockDim.x)
        .set_imm(&msg::request::imms::block_y, (uint64_t)blockDim.y)
        .set_imm(&msg::request::imms::block_z, (uint64_t)blockDim.z)
        .set_imm(&msg::request::imms::stream_id, stream_id)
        .set_cap(&msg::request::caps::continuation, resp);

    size_t offset = offsetof(msg::request::imms, kernel_args);
    for (size_t i = 0; i < pimpl.args_size.size(); i++) {
        auto size = pimpl.args_size[i];
        req.set_imm(offset, args[i], size);
        offset += size;
    }

    return req
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_cuda_response()
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}

core::future<void>
srv::Function::destroy()
{
    auto& pimpl = impl::Function::get(*this);
    auto self = pimpl.self.lock();
    return pimpl.destroy()
        // keep self alive
        .then([self](auto& fut) {
            return fut.get();
        });
}

core::future<void>
impl::Function::do_destroy()
{
    METHOD(Function, destroy);
    LOG_REQ(method)
        << " {}";

    // NOTE: kept alive by Function::destroy() and ~Function()
    auto self = this;

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_generic)
        .set_imm(&msg::request::imms::opcode, srv_wire::OP_DESTROY)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp)
        .unwrap()
        .then_cuda_response()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();
            CHECK_ARGS_EXACT();
        });
}
