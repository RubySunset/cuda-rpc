
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
                    // fractos::core::cap::request test;
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
