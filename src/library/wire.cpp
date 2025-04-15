#include <fractos/core/builder.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;


std::string
to_string(const core::cap::generic& obj)
{
    std::stringstream ss;

    if (obj.is_type(core::cap::memory_tag)) {
        ss << "cap::memory(" << obj.get_cid() << ")";
    } else if (obj.is_type(core::cap::request_tag)) {
        ss << "cap::request(" << obj.get_cid() << ")";
    } else if (obj.is_type(core::cap::endpoint_tag)) {
        ss << "cap::endpoint(" << obj.get_cid() << ")";
    } else {
        ss << "cap::null(" << obj.get_cid() << ")";
    }

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::get_driver_version::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    if (obj.imms_size() == 0) {
        ss << "imms=<empty>";
    } else {
        ss << "imms=<malformed: size=" << obj.imms_size() << ">";
    }

    if (obj.has_cap(&msg::caps_type::continuation)) {
        ss << " caps.continuation=" << ::to_string(obj.caps.continuation);
    } else {
        ss << " caps.continuation=<missing>";
    }

    if (obj.has_all_caps() and not obj.has_exactly_caps()) {
        ss << " caps=<malformed: count=" << obj.caps_count() << ">";
    }

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::get_driver_version::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    if (obj.has_imm(&msg::imms_type::error)) {
        auto error = static_cast<fractos::wire::error_type>(obj.imms.error.get());
        ss << "imms.error=" << fractos::wire::to_string(error);
    } else {
        ss << "imms.error=<missing>";
    }

    if (obj.has_imm(&msg::imms_type::value)) {
        ss << "imms.value=" << obj.imms.value;
    } else {
        ss << "imms.value=<missing>";
    }

    if (obj.has_all_imms() and not obj.has_exactly_imms()) {
        ss << " imms=<malformed: size=" << obj.imms_size() << ">";
    }

    if (obj.caps_count() == 0) {
        ss << "caps=<empty>";
    } else {
        ss << "caps=<malformed: count=" << obj.caps_count() << ">";
    }

    return ss.str();
}
