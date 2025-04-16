#include <fractos/core/builder.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


#define print_imm(name, func)                                           \
    if (obj.has_imm(&msg::imms_type:: name)) {                          \
        ss << " imms." #name "=" << func(obj.imms. name);               \
    } else {                                                            \
        ss << " imms." #name "=<missing>";                              \
    }

#define print_imm_identity(name)                                        \
    print_imm(name, [](auto& val){ return val; })

#define print_imm_error(name)                                           \
    print_imm(name, [](auto& val){                                      \
        return fractos::wire::to_string(static_cast<fractos::wire::error_type>(val.get())); })

#define print_extra_imm_error()                                         \
    if (obj.has_all_imms() and not obj.has_exactly_imms()) {            \
        ss << " imms=<malformed: size=" << obj.imms_size() << ">";      \
    }

#define print_empty_imms()                                              \
    if (obj.imms_size() == 0) {                                         \
        ss << " imms=<empty>";                                           \
    } else {                                                            \
        ss << " imms=<malformed: size=" << obj.imms_size() << ">";       \
    }

#define print_cap(name)                                                 \
    if (obj.has_cap(&msg::caps_type:: name)) {                          \
        ss << " caps." #name "=" << core::to_string(obj.caps. name);    \
    } else {                                                            \
        ss << " caps." #name "=<missing>";                              \
    }

#define print_extra_cap_error()                                         \
    if (obj.has_all_caps() and not obj.has_exactly_caps()) {            \
        ss << " caps=<malformed: count=" << obj.caps_count() << ">";    \
    }

#define print_empty_caps()                                              \
    if (obj.caps_count() == 0) {                                        \
        ss << " caps=<empty>";                                           \
    } else {                                                            \
        ss << " caps=<malformed: count=" << obj.caps_count() << ">";     \
    }

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::connect::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_empty_imms();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::connect::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_extra_imm_error();

    print_cap(connect);
    print_cap(get_driver_version);
    print_cap(make_device);
    print_cap(get_device);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::get_driver_version::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_empty_imms();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::get_driver_version::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_identity(value);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::init::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(flags);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::init::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}
