
#include <fractos/core/cap.hpp>
#include <fractos/wire/endian.hpp>
// #include <fractos/service/compute/cuda_msg.hpp>
// #include <fractos/service/compute/cuda.hpp>

namespace fractos::service::compute::cuda::message  {

    struct connect_service  {
        struct request {
            struct imms {
            } __attribute__((packed));
            struct caps {
                core::cap::request cont;
            };
        };

        struct response {
            struct imms {
                wire::endian::uint8_t error;
            } __attribute__((packed));
            struct caps {
                core::cap::request make_device;
            };
        };
    };

    struct destroy {
        struct request {
            struct imms {
            } __attribute__((packed));
            struct caps {
                core::cap::request cont;
            };
        };

        struct response {
            struct imms {
                wire::endian::uint8_t error;
            } __attribute__((packed));
            struct caps {
            };
        };
    };


} // namespace fractos::service::compute::cuda::message


namespace fractos::service::compute::cuda::message::service {

    struct make_device {
        struct request {
            struct imms {
                wire::endian::uint64_t value; // device_id
            } __attribute__((packed));
            struct caps {
                core::cap::request cont;
            };
        };

        struct response {
            struct imms {
                wire::endian::uint8_t error;
            } __attribute__((packed));
            struct caps {
                // core::cap::request make_cuda_context;
                core::cap::request destroy;
            };
        };
    };

    struct destroy {
        struct request {
            struct imms {
            } __attribute__((packed));
            struct caps {
                core::cap::request cont;
            };
        };

        struct response {
            struct imms {
                wire::endian::uint8_t error;
            } __attribute__((packed));
            struct caps {
            };
        };
    };

} // namespace fractos::service::compute::cuda::message::Service


namespace fractos::service::compute::cuda::message::device {

    struct make_cuda_context {
        struct request {
            struct imms {
                // wire::endian::uint64_t num; // ctx_id
            } __attribute__((packed));
            struct caps {
                core::cap::request cont;
                // wire::endian::uint64_t num;
            };
        };

        struct response {
            struct imms {
                wire::endian::uint8_t error;
            } __attribute__((packed));
            struct caps {
            };
        };
    };

    struct destroy {
        struct request {
            struct imms {
            } __attribute__((packed));
            struct caps {
                core::cap::request cont;
            };
        };

        struct response {
            struct imms {
                wire::endian::uint8_t error;
            } __attribute__((packed));
            struct caps {
            };
        };
    };


} // namespace fractos::service::compute::cuda::message::Device

