
#include <fractos/core/cap.hpp>
#include <fractos/wire/endian.hpp>
using namespace fractos;
// #include <fractos/service/compute/cuda_msg.hpp>
// #include <fractos/service/compute/cuda.hpp>
namespace service::compute::cuda::message{
  
    namespace cuda_service {
        struct make_cuda_device {
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
                    fractos::core::cap::request make_cuda_context;
                    fractos::core::cap::request destroy;
                };
            };
        };

        struct get_cuda_device {
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
                    fractos::core::cap::request make_cuda_context;
                    fractos::core::cap::request destroy;
                };
            };
        };
    }

    namespace cuda_device {
        struct make_cuda_context {
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
                    fractos::core::cap::request make_cumemalloc;
                    fractos::core::cap::request make_module_file;
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

    namespace cuda_context {

        struct make_cumemalloc {
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
                    // fractos::core::cap::cap::memory memory;
                    fractos::core::cap::request destroy;
                };
            };
        };

        struct make_module_file {
            struct request {
                struct imms {
                    // fractos::wire::endian::uint64_t virtual_device_id;
                    // fractos::wire::endian::uint64_t name; // transfer through string with uint64_t num = std::stoull(str);
                    fractos::wire::endian::uint64_t func_name_size;
                    char func_name[];
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

    namespace cuda_memory {

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

    namespace cuda_module {

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
    
    
    
}
