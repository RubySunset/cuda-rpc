#include <cuda_runtime.h>
#include <fractos/common/service/wire_impl.hpp>
#include <fractos/core/builder.hpp>
#include <fractos/service/compute/cuda_msg.hpp>
#include <fractos/wire/error.hpp>
#include <iomanip>

using namespace fractos;
namespace srv = fractos::service::compute::cuda;

#define print_imm_cuerror(name)                                         \
    print_imm(name, [](auto& val){                                      \
        const char* name;                                               \
        CHECK(cuGetErrorName((CUresult)val.get(), &name) == CUDA_SUCCESS); \
        return name;                                                    \
    })


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
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_properties::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Device::get_properties::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_imm_identity(data_size);
    print_imm_array(data, obj.imms_size() - sizeof(obj.imms));

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Device::destroy::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Device::destroy::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
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
    print_cap(generic);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::mem_get_info::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Context::mem_get_info::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_imm_identity(free);
    print_imm_identity(total);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::memset::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_imm_hex(addr);
    print_imm_identity(row_elems);
    print_imm_identity(row_pad);
    print_imm_identity(row_count);
    print_imm_identity(value);
    print_imm_identity(value_bytes);
    print_imm_identity(custream);
    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::memset::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::synchronize::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_empty_imms();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::synchronize::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::module_load_data::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(opcode);
    print_extra_imm_error();

    print_cap(continuation);
    print_cap(contents);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::module_load_data::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_imm_hex(cumodule);
    print_extra_imm_error();

    print_cap(generic);
    print_cap(get_function);
    print_cap(destroy);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::stream_create::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Context::stream_create::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_imm_hex(custream);
    print_extra_imm_error();

    print_cap(generic);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::event_create::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Context::event_create::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_extra_imm_error();

    print_cap(generic);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::destroy::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_empty_imms();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Context::destroy::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Stream::synchronize::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Stream::synchronize::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Stream::destroy::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Stream::destroy::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_extra_imm_error();

    print_empty_caps();

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
srv::wire::to_string(const core::receive_args<srv::wire::Device::ctx_create::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Device::ctx_create::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_extra_imm_error();

    print_cap(generic);
    print_cap(synchronize);
    print_cap(destroy);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Function::launch::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_identity(grid_x);
    print_imm_identity(grid_y);
    print_imm_identity(grid_z);
    print_imm_identity(block_x);
    print_imm_identity(block_y);
    print_imm_identity(block_z);
    print_imm_identity(custream);
    print_imm_array(kernel_args, obj.imms_size() - sizeof(obj.imms));

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Function::launch::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Function::destroy::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_extra_imm_error();

    print_cap(continuation);
    print_extra_cap_error();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Function::destroy::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Event::destroy::request>& obj)
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
srv::wire::to_string(const core::receive_args<srv::wire::Event::destroy::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;

    std::stringstream ss;

    print_imm_error(error);
    print_imm_cuerror(cuerror);
    print_extra_imm_error();

    print_empty_caps();

    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Memory::destroy::request>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;
    std::stringstream ss;
    print_empty_imms();
    print_cap(continuation);
    print_extra_cap_error();
    return ss.str();
}

std::string
srv::wire::to_string(const core::receive_args<srv::wire::Memory::destroy::response>& obj)
{
    using msg = std::remove_cvref_t<decltype(obj)>;
    std::stringstream ss;
    print_imm_error(error);
    print_imm_cuerror(cuerror);
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
