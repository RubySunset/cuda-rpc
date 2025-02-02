#pragma once

#include <fractos/core/future.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <glog/logging.h>
#include <any>
#include <cuda.h>


// #include "../include/device_service_msgs.hpp"
// #include "device_memory.hpp"
// #include "device_function.hpp"

namespace fractos::service::compute { namespace [[gnu::visibility("default")]] cuda {

        class service;
        class device;
        class context;
        class function;
        class stream;
        class kernel;
        class memory;
//event
//graph
//module

        struct error {
            error(CUresult code);

            const CUresult code;
        };

        /**
         * @brief Connect to the CUDA service with given name
         */
        [[nodiscard]] core::future<std::shared_ptr<service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::gns::service& gns, const std::string& name,
                     const std::chrono::microseconds& wait_time = std::chrono::seconds{0});

        [[nodiscard]] core::future<std::shared_ptr<service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::cap::request& service_req);

        /**
         * The service implicitly calls cuInit() when started.
         */
        class service {
        public:
            [[nodiscard]] core::future<std::shared_ptr<device>>
            make_device(uint64_t device_id);

            /**
             * @brief Destroy service connection, and all created objects
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~service();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        /**
         * @brief Wrapper for CUdevice operations
         */
        class device {
        public:

            /**
             * @brief Wrapper for cuCtxCreate_v4()
             */
            [[nodiscard]] core::future<context>
            make_context(std::vector<CUctxCreateParams>& params,  unsigned int flags);

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~device();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        /**
         * @brief Wrapper for CUcontext operations
         */
        class context {
        public:

            /**
             * @brief Wrapper for cuMemAlloc()
             */
            [[nodiscard]] core::future<memory>
            make_memory(size_t size);

            /**
             * @brief Wrapper for cuMemAllocHost()
             */
            [[nodiscard]] core::future<memory>
            make_memory_host(size_t size);

            /**
             * @brief Wrapper for cuMemAllocManaged()
             */
            [[nodiscard]] core::future<memory>
            make_memory_managed(size_t size, CUmemAttach_flags flags);

            /**
             * @brief Wrapper for cuCtxSynchronize()
             */
            [[nodiscard]] fractos::core::future<void>
            synchronize();

            /**
             * @brief Destroy context and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~context();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        class device_function { // CUfunction
        public:
            fractos::core::future<device_function> get_function(std::weak_ptr<CUmodule> _module);

            fractos::core::future<void> unregister_func();

            device_function();

            ~device_function();

        public:
            std::shared_ptr<device_function> _self;

            std::weak_ptr<CUmodule> _module;

        private:
    
            bool _unregistered;
        };


        class device_stream { // CUfunction
        public:
            fractos::core::future<device_stream> create_stream(std::weak_ptr<virtual_context> _ctx);

            fractos::core::future<void> destroy_kernel();

            device_stream();

            ~device_stream();

        public:
            std::shared_ptr<device_stream> _self;

            std::weak_ptr<virtual_context> _ctx;

        private:
    
            bool _destroyed;
        };


        class device_kernel { // CUfunction
        public:
            fractos::core::future<device_kernel> launch_kernel(std::weak_ptr<device_stream> _stream,
                                                               const std::unordered_map<std::string, std::any>& backend_arg);

            fractos::core::future<void> unregister_kernel();

            device_kernel();

            ~device_kernel();

        public:
            std::shared_ptr<device_kernel> _self;
    
            std::weak_ptr<device_stream> _stream;

        private:
    
            bool _unregistered;
        };


        class device_memory { // CUfunction
        public:
            fractos::core::future<CUdeviceptr> alloc_memory(size_t size, std::weak_ptr<virtual_context> _ctx, const std::unordered_map<std::string, std::any>& backend_arg); // dst ? 

            fractos::core::future<void> memcpyH2D( size_t size, std::weak_ptr<device_stream> _stream, 
                                                   const std::unordered_map<std::string, std::any>& backend_arg); // dst, src, size
    
            fractos::core::future<void> memcpyD2H( size_t size, std::weak_ptr<device_stream> _stream, 
                                                   const std::unordered_map<std::string, std::any>& backend_arg); // dst, src, size

            fractos::core::future<void> deallocate_memory();

            device_memory();

            ~device_memory();

        public:
            std::shared_ptr<device_memory> _self;
            std::shared_ptr<CUdeviceptr> _devptr;

            std::weak_ptr<device_stream> _stream;
            std::weak_ptr<virtual_context> _ctx

            private:
            bool _dealloc;
        };

    } // namespace cuda
} // namespace fractos::service::compute
