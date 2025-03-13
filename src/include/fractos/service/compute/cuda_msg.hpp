
#include <fractos/core/cap.hpp>
#include <fractos/wire/endian.hpp>
using namespace fractos;

namespace service::compute::cuda::wire{
  
    namespace Service {
        struct make_device {
            struct request {
                struct imms {
                    fractos::wire::endian::uint8_t value;
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
                    fractos::core::cap::request make_context;
                    fractos::core::cap::request destroy;
                };
            };
        };

        struct get_Device {
            struct request {
                struct imms {
                    fractos::wire::endian::uint8_t value;
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
                    fractos::core::cap::request make_context;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    namespace Device {
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
                    fractos::core::cap::request make_stream;
                    fractos::core::cap::request make_module_data; // make_module_file
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

    namespace Context {

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
                    fractos::wire::endian::uint64_t file_name_size;
                    char file_name[];
                } __attribute__((packed));
                struct caps {
                    fractos::core::cap::memory cuda_file;
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
                    fractos::wire::endian::uint64_t grid;
                    fractos::wire::endian::uint64_t block;
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
