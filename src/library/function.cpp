#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>

#include <common.hpp>
#include <function_impl.hpp>
#include <stream_impl.hpp>


// #include <fractos/service/compute/cuda_msg.hpp>
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
                         fractos::core::cap::request req_func_call,
                         fractos::core::cap::request req_func_destroy)
    :ch(ch)
    ,args_total_size(args_total_size)
    ,args_size(args_size)
    ,req_func_call(std::move(req_func_call))
    ,req_func_destroy(std::move(req_func_destroy))
{
}

srv::Function::Function(std::shared_ptr<void> pimpl, std::string func_name)
    :_pimpl(pimpl)
{
}

srv::Function::~Function()
{
    destroy()
        .then([pimpl=this->_pimpl](auto& fut) {
            fut.get();
        })
        .as_callback();
}

core::future<void>
srv::Function::launch(dim3 gridDim, dim3 blockDim, const void** args,
                      size_t sharedMemBytes,
                      std::optional<std::reference_wrapper<Stream>> stream)
{
    METHOD(Function, call);
    LOG_REQ(method)
        << " {}";

    auto& pimpl = impl::Function::get(*this);

    uint32_t stream_id = 0;
    if (stream) {
        auto& stream_pimpl = impl::Stream::get(stream->get());
        stream_id = stream_pimpl.id;
    }

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    auto req = pimpl.ch->make_request_builder<msg::request>(pimpl.req_func_call)
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
        .then([this, self=pimpl.self.lock()](auto& fut) {
            auto [ch, args] = fut.get();

            LOG_RES_PTR(method, self)
                << wire::to_string(*args);
            CHECK_ARGS_EXACT();
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}

// template <size_t N, class T, class... Args>
// static inline void append_call_arg(size_t &offset, size_t &count, T &req,
//                                     std::tuple<Args...> args) {
//     if constexpr (N < std::tuple_size<decltype(args)>()) {
//     auto size = sizeof(std::get<N>(args));
//     using msg = ::service::compute::cuda::wire::Function::call;
//     msg::kernel_arg_info arg_info;
//     arg_info.size = size;
//     req.set_imm(offset, &arg_info, sizeof(arg_info));
//     offset += sizeof(arg_info);
//     req.set_imm(offset, &std::get<N>(args), size);
//     offset += sizeof(std::get<N>(args));
//     count++;

//     append_call_arg<N + 1>(offset, count, req, args);
//     }
// }

// template<class... Args>
// core::future<void> Function::call(Args&&... ker_args) {
// // core::future<void> Function::call(std::pair<size_t, size_t>& gpu_grid, Args&&... ker_args) {
//     using msg = ::service::compute::cuda::wire::Function::call;

//     DVLOG(logging::SERVICE) << "Function::call <-";

//     auto& pimpl = impl::Function::get(*this);

//     // auto kargs = std::make_tuple<Args...>(std::forward<Args>(ker_args)...);

//     auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
//     auto req =  pimpl.ch->make_request_builder<msg::request>(pimpl.req_func_call)
//         // .set_imm(&msg::request::imms::grid, (uint64_t)gpu_grid.first)
//         // .set_imm(&msg::request::imms::block, (uint64_t)gpu_grid.second)
//         .set_cap(&msg::request::caps::continuation, resp);
//         // .set_cap(&msg::request::caps::continuation_success, resp)
//         // .set_cap(&msg::request::caps::continuation_failure, resp)

//     size_t cur_offset = offsetof(msg::request::imms, block) + sizeof(uint64_t);
//     size_t args_num = 0;
//     // append_call_arg<0>(cur_offset, args_num, req, kargs);
//     req.set_imm(&msg::request::imms::args_num, (uint64_t)args_num);
    

//     return req
//         // .set_cap(&msg::request::caps::continuation, resp)
//         .on_channel()
//         .invoke(resp) // wait for handle_sync
//         .unwrap()
//         .then([](auto& fut) {
//             auto [ch, args] = fut.get();

//             if (not args->has_exactly_args()) {
//                 // throw core::other_error("invalid response format for Function::call");
//                 DVLOG(logging::SERVICE) << "Function::call ->"
//                                 << " error=OTHER args";
//             }

//             DVLOG(logging::SERVICE) << "Function::call ->"
//                                     << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
//             wire::error_raise_exception_maybe(args->imms.error);
//         });
// }



core::future<void>
srv::Function::destroy()
{
    auto& pimpl = impl::Function::get(*this);
    return pimpl.destroy();
}

core::future<void>
impl::Function::do_destroy()
{
    using msg = ::service::compute::cuda::wire::Function::func_destroy;

    DVLOG(logging::SERVICE) << "Function::func_destroy <-";

    auto resp = ch->make_response_builder<msg::response>(ch->get_default_endpoint());
    return ch->make_request_builder<msg::request>(req_func_destroy)
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_destroy
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Function::destroy");
                DVLOG(logging::SERVICE) << "Function::func_destroy ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Function::func_destroy ->"
                                    << " error=" << fractos::wire::to_string((fractos::wire::error_type)args->imms.error.get());
            fractos::wire::error_raise_exception_maybe(args->imms.error);
        });
}
