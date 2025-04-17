#pragma once

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

        enum generic_opcode : uint64_t {
            OP_GET_DRIVER_VERSION,
            OP_INIT,

            OP_DEVICE_GET,
            OP_DEVICE_GET_COUNT,

            OP_MODULE_GET_LOADING_MODE,

            OP_INVALID = std::numeric_limits<uint64_t>::max()
        };

        using generic = wire::generic;

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

    std::string to_string(const core::receive_args<Service::connect::request>& req);
    std::string to_string(const core::receive_args<Service::connect::response>& resp);

    std::string to_string(const core::receive_args<Service::get_driver_version::request>& req);
    std::string to_string(const core::receive_args<Service::get_driver_version::response>& resp);

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
                    fractos::core::cap::request make_context;
                    fractos::core::cap::request destroy;
                };
            };
        };

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

    std::string to_string(const core::receive_args<Service::device_get::request>& req);
    std::string to_string(const core::receive_args<Service::device_get::response>& resp);

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
            OP_GET_NAME,

            OP_INVALID = std::numeric_limits<uint64_t>::max()
        };

        using generic = wire::generic;

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

        struct make_context {
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

    std::string to_string(const core::receive_args<Device::get_name::request>& req);
    std::string to_string(const core::receive_args<Device::get_name::response>& resp);


    namespace Context {

        struct make_memory_rpc_test {
            struct request {
                struct imms {
                    // fractos::wire::endian::uint64_t virtual_device_id;
                    fractos::wire::endian::uint64_t size; // unsigned int
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
                    // fractos::core::cap::memory memory;
                    // fractos::core::cap::request destroy;
                };
            };
        };
        struct make_memory {
            struct request {
                struct imms {
                    // fractos::wire::endian::uint64_t virtual_device_id;
                    fractos::wire::endian::uint64_t size; // unsigned int
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::request continuation; 
                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                    fractos::wire::endian::uint64_t address;
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::memory memory;
                    fractos::core::cap::request destroy;
                };
            };
        };

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
                    // fractos::core::cap::cap::memory memory;
                    fractos::core::cap::request get_function;
                    fractos::core::cap::request destroy;
                };
            };
        };
        
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
                } __attribute__ ((packed));
                struct caps {
                    fractos::core::cap::request call;
                    fractos::core::cap::request func_destroy; 
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
        struct call {
            struct request {
                struct imms {
                    fractos::wire::endian::uint64_t args_num;
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
                    // fractos::core::cap::request continuation_success; 
                    // fractos::core::cap::request continuation_failure;

                };
            };
            struct response {
                struct imms {
                    fractos::wire::endian::uint8_t error;
                } __attribute__ ((packed));
                struct caps {
                };
            };
            struct kernel_arg_info {
                fractos::wire::endian::uint64_t size;
                char value[];
            };

        };

        struct func_destroy {
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

}
