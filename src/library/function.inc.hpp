
#include <utility>

#include <fractos/wire/error.hpp>
#include <fractos/core/future.hpp>
#include <fractos/logging.hpp>
#include <fractos/service/compute/cuda.hpp>
#include <fractos/service/compute/cuda_msg.hpp>


// #include </home/mingxuanyang/fractos/experiments/deps/service-compute-cuda/src/library/function_impl.hpp>
#include <function_impl.hpp>

// #include <fractos/service/compute/cuda_msg.hpp>
using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace impl;

template <size_t N, class T, class... Args>
static inline void append_call_arg(size_t &offset, size_t &count, T &req,
                                    std::tuple<Args...> args) {
    if constexpr (N < std::tuple_size<decltype(args)>()) {
    auto size = sizeof(std::get<N>(args));
    using msg = ::service::compute::cuda::wire::Function::call;
    msg::kernel_arg_info arg_info;
    arg_info.size = size;
    req.set_imm(offset, &arg_info, sizeof(arg_info));
    offset += sizeof(arg_info);
    req.set_imm(offset, &std::get<N>(args), size);
    offset += sizeof(std::get<N>(args));
    count++;

    append_call_arg<N + 1>(offset, count, req, args);
    }
}

template<class... Args>
core::future<void> Function::call(std::pair<size_t, size_t>& gpu_grid, Args&&... ker_args) {
    using msg = ::service::compute::cuda::wire::Function::call;

    DVLOG(logging::SERVICE) << "Function::call <-";

    auto& pimpl = Function_impl::get(*this);

    auto kargs = std::make_tuple<Args...>(std::forward<Args>(ker_args)...);

    auto resp = pimpl.ch->make_response_builder<msg::response>(pimpl.ch->get_default_endpoint());
    auto req =  pimpl.ch->make_request_builder<msg::request>(pimpl.req_func_call)
        .set_imm(&msg::request::imms::grid, (uint64_t)gpu_grid.first)
        .set_imm(&msg::request::imms::block, (uint64_t)gpu_grid.second)
        .set_cap(&msg::request::caps::continuation, resp);
        // .set_cap(&msg::request::caps::continuation_success, resp)
        // .set_cap(&msg::request::caps::continuation_failure, resp)

    size_t cur_offset = offsetof(msg::request::imms, block) + sizeof(uint64_t);
    size_t args_num = 0;
    append_call_arg<0>(cur_offset, args_num, req, kargs);
    req.set_imm(&msg::request::imms::args_num, (uint64_t)args_num);
    

    return req
        .set_cap(&msg::request::caps::continuation, resp)
        .on_channel()
        .invoke(resp) // wait for handle_sync
        .unwrap()
        .then([](auto& fut) {
            auto [ch, args] = fut.get();

            if (not args->has_exactly_args()) {
                // throw core::other_error("invalid response format for Function::call");
                DVLOG(logging::SERVICE) << "Function::call ->"
                                << " error=OTHER args";
            }

            DVLOG(logging::SERVICE) << "Function::call ->"
                                    << " error=" << wire::to_string((wire::error_type)args->imms.error.get());
            wire::error_raise_exception_maybe(args->imms.error);
        });
}