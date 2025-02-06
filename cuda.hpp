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
        class clientService;
        class device;
        class context;
        class kernel;
        // non - context-agnostic
        class function;
        class stream;
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
            [[nodiscard]] core::future<std::shared_ptr<clientService>>
            make_client(std::shared_ptr<core::channel> ch,
                     core::gns::service& gns, const std::string& name,
                     const std::chrono::microseconds& wait_time = std::chrono::seconds{0});

            [[nodiscard]] core::future<std::shared_ptr<clientService>>
            make_client(std::shared_ptr<core::channel> ch,
                     core::cap::request& service_req);


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
         * The client service implicitly calls cuInit() when started.
         */
        class clientService {
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
             * @brief Wrapper for cuLibraryLoadFromFile()
             */
            [[nodiscard]] core::future<std::shared_ptr<kernel>>
            make_Kernel(const char* lib_fileName,  const char* function_name); // cuLibraryLoadFromFile from ptx file

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
         * @brief Wrapper for CUkernel operations
         */
        class kernel {
        public:

            /**
             * @brief Wrapper for cuKernelGetFunction()
             */
            [[nodiscard]] core::future<function>
            make_function(context ctx); // cuKernelGetFunction 

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~kernel();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        

        /**
         * @brief Wrapper for CUcontext operations
         */
        class context {
        public:

            /**
             * @brief Wrapper for cuModuleLoad ()
             */
            [[nodiscard]] core::future<module>
            make_load_module( const char* fname);

            /**
             * @brief Wrapper for cuStreamCreate()
             */
            [[nodiscard]] core::future<stream>
            make_stream( unsigned int  Flags);

            /**
             * @brief Wrapper for cuMemAlloc()
             */
            [[nodiscard]] core::future<device_memory>
            make_memory(size_t size);

            /**
             * @brief Wrapper for cuMemAllocHost()
             */
            [[nodiscard]] core::future<device_memory>
            make_memory_host(size_t size);

            /**
             * @brief Wrapper for cuMemAllocManaged()
             */
            [[nodiscard]] core::future<device_memory>
            make_memory_managed(size_t size, CUmemAttach_flags flags);


            /**
             * @brief Wrapper for cuMemAddressReserve()
             */
            [[nodiscard]] core::future<virtual_memory>
            make_vmemory_space(size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);

            
           


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

        class module{
        public:

            /**
             * @brief Wrapper for cuModuleGetFunction ()
             */
            [[nodiscard]] fractos::core::future<function>
            get_function(const char* name);

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~module();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        }

        class function{
        public:

             /**
             * @brief Wrapper for cuLaunchKernel()
             */
            [[nodiscard]] core::future<void>
            launch_kernel(stream stream, const std::unordered_map<std::string, std::any>& backend_arg);

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~function();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        }

        class stream{
        public:

            /**
             * @brief Wrapper for cuStreamSynchronize()
             */
            [[nodiscard]] fractos::core::future<void>
            synchronize();

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~stream();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        }

        class device_memory{
        public:

            /**
             * @brief Wrapper for cuMemcpy()
             */
            [[nodiscard]] core::future<void>
            memory_copy(size_t size, CUstream stream, const std::unordered_map<std::string, std::any>& backend_arg); // dst src

            /**
             * @brief Wrapper for cuMemFree()
             */
            [[nodiscard]] core::future<void>
            memory_free(size_t size);

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~device_memory();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        }

        class virtual_memory{
        public:

            /**
             * @brief Wrapper for cuMemCreate()
             */
            [[nodiscard]] core::future<CUmemGenericAllocationHandle>
            make_vmemory_physical_handle( size_t size, const CUmemAllocationProp* prop, unsigned long long flags);

            /**
             * @brief Wrapper for cuMemMap()
             */
            [[nodiscard]] core::future<void>
            vmemory_virtual_mapping(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flag);

            /**
             * @brief Wrapper for cuMemUnmap ( CUdeviceptr ptr, size_t size )
             */
            [[nodiscard]] core::future<void>
            vmemory_virtual_unmap(CUdeviceptr ptr, size_t size);

            /**
             * @brief Wrapper for cuMemSetAccess()
             */
            [[nodiscard]] core::future<void>
            vmemory_set_access( CUdeviceptr ptr, size_t size, const CUmemAccessDesc* desc, size_t count);

            /**
             * @brief Wrapper for cuMemRelease() // cuMemCreate
             */
            [[nodiscard]] core::future<void>
            vmemory_free_physical(CUmemGenericAllocationHandle handle);
            

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~virtual_memory();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        }

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

    } // namespace cuda
} // namespace fractos::service::compute
