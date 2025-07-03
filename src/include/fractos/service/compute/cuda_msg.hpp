#pragma once

#include <cuda.h>
#include <fractos/core/builder.hpp>
#include <fractos/core/cap.hpp>
#include <fractos/wire/endian.hpp>

namespace fractos::service::compute::cuda::wire {

    struct generic {
        struct request {
            struct imms {
                fractos::wire::endian::uint64_t opcode;
            } __attribute__((packed));
            struct caps {
                fractos::core::cap::request continuation;
            };
        };
        struct response {
            struct imms {
                fractos::wire::endian::uint8_t error;
            } __attribute__ ((packed));
            struct caps {
            };
        };
    };

    namespace Service {

        struct connect {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request connect;
                    fractos::core::cap::request generic;
                };
            };
        };

    }

    std::string to_string(const core::receive_args<Service::connect::request>& req);
    std::string to_string(const core::receive_args<Service::connect::response>& resp);

    namespace Service {

        enum generic_opcode : uint64_t {
            OP_GET_DRIVER_VERSION,
            OP_INIT,

            OP_DEVICE_GET,
            OP_DEVICE_GET_COUNT,

            OP_MODULE_GET_LOADING_MODE,

            OP_INVALID = std::numeric_limits<uint64_t>::max()
        };

        using generic = wire::generic;

    }

    namespace Service {
        struct get_driver_version {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t value;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Service::get_driver_version::request>& req);
    std::string to_string(const core::receive_args<Service::get_driver_version::response>& resp);

    namespace Service {
        struct init {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint64_t flags;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Service::init::request>& req);
    std::string to_string(const core::receive_args<Service::init::response>& resp);


    namespace Service {
        struct device_get {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint8_t ordinal;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t device;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request generic;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Service::device_get::request>& req);
    std::string to_string(const core::receive_args<Service::device_get::response>& resp);

    namespace Service {
        struct device_get_count {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t count;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };

    }

    std::string to_string(const core::receive_args<Service::device_get_count::request>& req);
    std::string to_string(const core::receive_args<Service::device_get_count::response>& resp);


    namespace Service {
        struct module_get_loading_mode {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t mode;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Service::module_get_loading_mode::request>& req);
    std::string to_string(const core::receive_args<Service::module_get_loading_mode::response>& resp);


    namespace Device {

        enum generic_opcode : uint64_t {
            OP_GET_ATTRIBUTE,
            OP_GET_NAME,
            OP_GET_UUID,
            OP_TOTAL_MEM,
            OP_CTX_CREATE,

            OP_INVALID = std::numeric_limits<uint64_t>::max()
        };

        using generic = wire::generic;
    }

    namespace Device {
        struct get_attribute {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint64_t attrib;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t pi;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Device::get_attribute::request>& req);
    std::string to_string(const core::receive_args<Device::get_attribute::response>& resp);

    namespace Device {
        struct get_name {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t len;
                    char name[];
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Device::get_name::request>& req);
    std::string to_string(const core::receive_args<Device::get_name::response>& resp);

    namespace Device {
        struct get_uuid {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint8_t uuid[16];
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Device::get_uuid::request>& req);
    std::string to_string(const core::receive_args<Device::get_uuid::response>& resp);

    namespace Device {
        struct total_mem {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t bytes;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Device::total_mem::request>& req);
    std::string to_string(const core::receive_args<Device::total_mem::response>& resp);

    namespace Device {
        struct ctx_create {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint32_t flags; // unsigned int
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t cuerror;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request generic;
                    fractos::core::cap::request make_memory;
                    fractos::core::cap::request make_memory_rpc_test;
                    fractos::core::cap::request make_stream;
                    fractos::core::cap::request make_event;
                    fractos::core::cap::request make_module_data; //make_module_data; // 
                    fractos::core::cap::request synchronize;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Device::ctx_create::request>& req);
    std::string to_string(const core::receive_args<Device::ctx_create::response>& resp);

    namespace Device {
        struct destroy {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Device::destroy::request>& req);
    std::string to_string(const core::receive_args<Device::destroy::response>& resp);


    namespace Context {

        enum generic_opcode : uint64_t {
            OP_GET_API_VERSION,
            OP_GET_LIMIT,
            OP_MEM_ALLOC,

            OP_INVALID = std::numeric_limits<uint64_t>::max()
        };

        using generic = wire::generic;

    }

    namespace Context {
        struct get_api_version {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t version;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::get_api_version::request>& req);
    std::string to_string(const core::receive_args<Context::get_api_version::response>& resp);

    namespace Context {
        struct get_limit {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint64_t limit;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t value;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::get_limit::request>& req);
    std::string to_string(const core::receive_args<Context::get_limit::response>& resp);

    namespace Context {
        struct mem_alloc {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint64_t size;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t cuerror;
                    fractos::wire::endian::uint64_t address;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::memory memory;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::mem_alloc::request>& req);
    std::string to_string(const core::receive_args<Context::mem_alloc::response>& resp);

    namespace Context {
        struct make_stream {
            struct request {
                struct imms {
                    fractos::wire::endian::uint32_t flags; // unsigned int
                    fractos::wire::endian::uint32_t stream_id;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request synchronize;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::make_stream::request>& req);
    std::string to_string(const core::receive_args<Context::make_stream::response>& resp);

    namespace Context {
        struct make_event {
            struct request {
                struct imms {
                    fractos::wire::endian::uint32_t flags; // unsigned int
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                    // fractos::core::cap::request record;
                    // fractos::core::cap::request synchronize;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::make_event::request>& req);
    std::string to_string(const core::receive_args<Context::make_event::response>& resp);

    namespace Context {
        struct make_module_file {
            struct request {
                struct imms {
                    // fractos::wire::endian::uint64_t virtual_device_id;
                    // fractos::wire::endian::uint64_t name; // transfer through string with uint64_t num = std::stoull(str);
                    fractos::wire::endian::uint64_t file_name_size;
                    char file_name[];
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    // fractos::wire::endian::uint64_t address;
                } __attribute__ ((packed));
                struct caps {
                    // fractos::core::cap::cap::memory memory;
                    fractos::core::cap::request get_function;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    namespace Context {
        struct make_module_data {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t module_id;
                    // char file_name[];
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                    fractos::core::cap::memory cuda_file;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    // fractos::wire::endian::uint64_t address;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request generic;
                    // fractos::core::cap::cap::memory memory;
                    fractos::core::cap::request get_function;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::make_module_data::request>& req);
    std::string to_string(const core::receive_args<Context::make_module_data::response>& resp);

    namespace Context {
        struct synchronize {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::synchronize::request>& req);
    std::string to_string(const core::receive_args<Context::synchronize::response>& resp);

    namespace Context {
        struct destroy {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Context::destroy::request>& req);
    std::string to_string(const core::receive_args<Context::destroy::response>& resp);

    namespace Stream {

        struct synchronize {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };

        struct destroy {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    namespace Event {

        // struct synchronize {
        //     struct request {
        //         struct imms {
        //         } __attribute__((packed));
        //         struct caps {
        //             fractos::core::cap::request continuation;
        //         };
        //     };
        //     struct response {
        //         struct imms {
        //             fractos::wire::endian::uint8_t error;
        //         } __attribute__ ((packed));
        //         struct caps {
        //         };
        //     };
        // };

    }

    namespace Event {
        struct destroy {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Event::destroy::request>& req);
    std::string to_string(const core::receive_args<Event::destroy::response>& resp);


    namespace Memory {

        struct destroy {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    namespace Module {

        enum generic_opcode : uint64_t {
            OP_GET_GLOBAL,

            OP_INVALID = std::numeric_limits<uint64_t>::max()
        };

        using generic = wire::generic;
    }

    namespace Module {
        struct get_global {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint64_t name_size;
                    char name[];
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t dptr;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Module::get_global::request>& req);
    std::string to_string(const core::receive_args<Module::get_global::response>& resp);

    namespace Module {
        struct get_function {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t func_name_size;
                    char func_name[]; // unsigned int
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t nargs;
                    fractos::wire::endian::uint64_t arg_size[];
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request generic;
                };
            };
        };

        struct destroy {
            struct request {
                struct imms {
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    namespace Function {

        enum generic_opcode : uint64_t {
            OP_LAUNCH,
            OP_DESTROY,
            OP_INVALID = std::numeric_limits<uint64_t>::max()
        };

        using generic = wire::generic;
    }

    namespace Function {
        struct launch {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                    fractos::wire::endian::uint64_t grid_x;
                    fractos::wire::endian::uint64_t grid_y;
                    fractos::wire::endian::uint64_t grid_z;
                    fractos::wire::endian::uint64_t block_x;
                    fractos::wire::endian::uint64_t block_y;
                    fractos::wire::endian::uint64_t block_z;
                    fractos::wire::endian::uint32_t stream_id;
                    char kernel_args[];
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t cuerror;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Function::launch::request>& req);
    std::string to_string(const core::receive_args<Function::launch::response>& resp);

    namespace Function {
        struct destroy {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t opcode;
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation;
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t cuerror;
                } __attribute__ ((packed));
                struct caps {
                };
            };
        };
    }

    std::string to_string(const core::receive_args<Function::destroy::request>& req);
    std::string to_string(const core::receive_args<Function::destroy::response>& resp);


    std::string to_string(CUuuid uuid);
}
