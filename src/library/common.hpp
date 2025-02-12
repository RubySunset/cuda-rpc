#pragma once

#include <fractos/logging.hpp>


// Some macros to help logging call/return of the exported methods

#define LOG_OP(func)                                            \
    VLOG(fractos::logging::SERVICE) << func
#define LOG_METHOD(method)                                      \
    VLOG(fractos::logging::SERVICE) << to_string(*this) << "::" << method
#define LOG_REQ(method)                                         \
    LOG_METHOD(method) << " <-"
#define LOG_RES(method)                                         \
    LOG_METHOD(method) << " ->"
#define LOG_METHOD_PTR(method, ptr)                                     \
    VLOG(fractos::logging::SERVICE) << to_string(*ptr) << "::" << method
#define LOG_REQ_PTR(method, ptr)                \
    LOG_METHOD_PTR(method, ptr) << " <-"
#define LOG_RES_PTR(method, ptr)                \
    LOG_METHOD_PTR(method, ptr) << " ->"
