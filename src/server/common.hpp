#pragma once

#include <fractos/logging.hpp>


#define LOG_OP(func)                            \
    VLOG(fractos::logging::SERVICE) << func
#define LOG_METHOD(method)                      \
    VLOG(fractos::logging::SERVICE) << to_string(*this) << "::" << method
#define LOG_REQ(method)                         \
    LOG_METHOD(method) << " ->"
#define LOG_RES(method)                         \
    LOG_METHOD(method) << " <-"
#define LOG_METHOD_PTR(method, ptr)                                     \
    VLOG(fractos::logging::SERVICE) << to_string(*ptr) << "::" << method
#define LOG_REQ_PTR(method, ptr)                \
    LOG_METHOD_PTR(method, ptr) << " ->"
#define LOG_RES_PTR(method, ptr)                \
    LOG_METHOD_PTR(method, ptr) << " <-"

// Helper for cases where we have no response Request to send an error to

#define as_callback_log_ignore_error(msg)                               \
    then([](auto& fut) {                                                \
             try {                                                      \
                  (void)fut.get();                                      \
              } catch (const std::runtime_error& e) {                   \
                  LOG(ERROR) << msg << ": " << e.what();                \
              }                                                         \
         })                                                             \
    .as_callback()

template<class T>
struct receive_args_base_type
{
    using type = std::remove_cvref_t<T>::element_type::base_type;
};

#define METHOD(cls, name)                                               \
    static const std::string method = "handle_" #name;                  \
    using msg = srv::wire:: cls :: name;                                \
    {                                                                   \
        using args_type = receive_args_base_type<decltype(args)>::type; \
        static_assert(std::is_same<msg::request, args_type>::value);    \
    }

#define CHECK_ARGS_COND(cond)                                           \
    if (not (cond)) {                                                   \
        LOG_RES(method) << " error=ERR_OTHER";                          \
        reqb_cont                                                       \
            .set_imm(&msg::response::imms::error, wire::ERR_OTHER)      \
            .on_channel()                                               \
            .invoke()                                                   \
            .as_callback_log_ignore_error("[error] failed to invoke continuation, ignoring"); \
        return;                                                         \
    }

#define CHECK_ARGS_EXACT()                                              \
    CHECK_ARGS_COND(args->has_exactly_args())

#define CHECK_IMMS_EXACT()                                              \
    CHECK_ARGS_COND(args->has_exactly_imms())

#define CHECK_CAPS_EXACT()                                              \
    CHECK_ARGS_COND(args->has_exactly_caps())

#define CHECK_IMMS_ALL()                                                \
    CHECK_ARGS_COND(args->has_all_imms())

#define CHECK_CAPS_ALL()                                                \
    CHECK_ARGS_COND(args->has_all_caps())
