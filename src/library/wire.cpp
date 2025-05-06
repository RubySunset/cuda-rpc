#include <cuda_runtime.h>
#include <fractos/core/builder.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <iomanip>

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

#define print_imm_hex(name)                                             \
    print_imm(name, [](auto& val){ std::stringstream ss; ss << "0x" << std::hex << val.get(); return ss.str(); })

#define print_imm_error(name)                                           \
    print_imm(name, [](auto& val){                                      \
        return fractos::wire::to_string(static_cast<fractos::wire::error_type>(val.get())); })

#define print_imm_cuerror(name)                                         \
    print_imm(name, [](auto& val){                                      \
        return cudaGetErrorName((cudaError)val.get()); })

#define print_imm_string(name_str, name_len)                            \
    if (obj.imms_size() >= offsetof(msg::imms_type, name_str) and       \
        obj.has_imm(&msg::imms_type:: name_len)) {                      \
        ss << " imms." #name_str "=\"" + std::string(obj.imms. name_str, obj.imms. name_len) + "\""; \
    } else {                                                            \
        ss << " imms." #name_str "=<missing>";                          \
    }

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
    print_cap(generic);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::get_driver_version::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

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

    print_imm_identity(opcode);
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


std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::device_get::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_imm(ordinal, [](auto& val){ return (int)val.get(); });
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::device_get::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm(device, [](auto& val){ return (int)val.get(); });
    print_extra_imm_error();

    print_cap(generic);
    print_cap(make_context);
    print_cap(destroy);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::device_get_count::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::device_get_count::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_identity(count);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}


std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::module_get_loading_mode::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Service::module_get_loading_mode::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_identity(mode);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}


std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_attribute::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_imm_identity(attrib);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_attribute::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_identity(pi);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_name::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_name::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_identity(len);
    if (obj.has_all_imms()) {
        ss << " imms.name=\"" << std::string(obj.imms.name, obj.imms.len) << "\"";
    }

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_uuid::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_uuid::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    ss << " imms.uuid=" << to_string(*(CUuuid*)&obj.imms.uuid);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::total_mem::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::total_mem::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_identity(bytes);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::get_api_version::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::get_api_version::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_identity(version);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::get_limit::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_imm_identity(limit);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::get_limit::response>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Context::mem_alloc::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_imm_identity(size);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::mem_alloc::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_imm_hex(address);
    print_extra_imm_error();

    print_cap(memory);
    print_cap(destroy);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Module::get_global::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_imm_identity(name_size);
    print_imm_string(name, name_size);

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Module::get_global::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_hex(dptr);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Function::call::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(grid_x);
    print_imm_identity(grid_y);
    print_imm_identity(grid_z);
    print_imm_identity(block_x);
    print_imm_identity(block_y);
    print_imm_identity(block_z);
    print_imm_identity(stream_id);

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Function::call::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(CUuuid uuid)
{
    std::stringstream ss;

    for (size_t i = 0; i < sizeof(uuid); i++) {
        auto elem = (unsigned)(uuid.bytes[i]) & 0x0ff;
        ss << std::hex << std::setfill('0') << std::setw(2) << elem;
    }

    return ss.str();
}
