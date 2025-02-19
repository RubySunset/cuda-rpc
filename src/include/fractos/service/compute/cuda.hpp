#pragma once

#include <any>
#include <cuda.h>
#include <fractos/core/future.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/gns.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <fractos/service/compute/cuda_msg.hpp>


namespace fractos::service::compute { namespace [[gnu::visibility("default")]] cuda {

        class Service;
        class Device;
        class Context;
        class Module;
        class Function;
        class Stream;
        class Memory;
        class MemoryReservation;
        class MemoryAllocation;
        //event
        //graph
        //module

        struct no_service_error : public std::runtime_error {
            no_service_error(const std::string& what);
        };
    
    

        struct ErrorChecker {
            ErrorChecker();
            // ErrorChecker(CUresult err,  const char *file, const int line);

            ErrorChecker(CUresult err, const std::string& file, int line);

        private:
            void handleError(CUresult err, const std::string& file, int line);
            const CUresult err;
        };


        /**
         * @brief Connect to the CUDA service with given name
         */
        [[nodiscard]] core::future<std::unique_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::gns::service& gns, const std::string& name,
                     const std::chrono::microseconds& wait_time = std::chrono::seconds{0});

        [[nodiscard]] core::future<std::unique_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::cap::request& service_req);
        
                     

        /**
         * The service implicitly calls cuInit() when started.
         */
        class Service {
        public:
            

            /**
             * @brief Wrapper for cuDeviceGet()
             */
            [[nodiscard]] core::future<std::shared_ptr<Device>>
            make_cuda_device(std::shared_ptr<core::channel> ch, uint64_t value);


            // [[nodiscard]] core::future<void>
            // make_service(std::string name);

            /**
             * @brief Destroy service connection, and all created objects
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            // Service();
            ~Service();
            // // NOTE: not for public use
            // Service(std::shared_ptr<void> pimpl);
            // Service(std::string name);
            std::shared_ptr<void> _pimpl;
        };

        std::string to_string(const Service& obj);

        /**
         * @brief Wrapper for CUdevice operations
         */
        class Device {
        public:

            /**
             * @brief Wrapper for cuCtxCreate_v4()
             */
            [[nodiscard]] core::future<std::shared_ptr<Context>>
            make_context(const std::vector<CUctxCreateParams>& params,  unsigned int flags);

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~Device();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };


        /**
         * @brief Wrapper for CUcontext operations
         */
        class Context {
        public:

            /**
             * @brief Wrapper for cuModuleLoad()
             */
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            make_module(const std::string file_path);

            /**
             * @brief Wrapper for cuModuleLoadData()
             */
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            make_module(const core::cap::memory& contents);

            /**
             * @brief Wrapper for cuStreamCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<Stream>>
            make_stream(CUstream_flags flags);

            /**
             * @brief Wrapper for cuMemAlloc()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory(size_t size);

            /**
             * @brief Wrapper for cuMemAllocHost()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory_host(size_t size);

            /**
             * @brief Wrapper for cuMemAllocManaged()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory_managed(size_t size, CUmemAttach_flags flags);

            /**
             * @brief Wrapper for cuMemAddressReserve()
             */
            [[nodiscard]] core::future<std::shared_ptr<MemoryReservation>>
            make_memory_reservation(size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);

            /**
             * @brief Wrapper for cuMemCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<MemoryAllocation>>
            make_memory_allocation(size_t size, const std::vector<CUmemAllocationProp>& props, unsigned long long flags);

            /**
             * @brief Wrapper for cuMemMap()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory(MemoryReservation& reservation, MemoryAllocation& allocation, unsigned long long flags);

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
            ~Context();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        /**
         * @brief Wrapper for CUmodule operations
         */
        class Module {
        public:

            /**
             * @brief Wrapper for cuModuleGetFunction ()
             */
            [[nodiscard]] fractos::core::future<std::shared_ptr<Function>>
            make_function(const std::string name);

            /**
             * @brief Destroy module and all its functions
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~Module();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        /**
         * @brief Wrapper for CUfunction operations
         */
        class Function{
        public:

             /**
             * @brief Wrapper for cuLaunchKernel()
             */
            template<class... Args>
            [[nodiscard]] core::future<void>
            call(Stream& stream, const std::tuple<size_t, size_t, size_t>& grid, Args&&... args);

            /**
             * @brief Destroy function
             *
             * @todo what about pending launches?
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~Function();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        class Stream{
        public:

            [[nodiscard]] fractos::core::future<void>
            copy(const Memory& src, core::cap::memory& dst);

            [[nodiscard]] fractos::core::future<void>
            copy(const core::cap::memory& src, Memory& dst);

            /**
             * @brief Wrapper for cuStreamSynchronize()
             */
            [[nodiscard]] fractos::core::future<void>
            synchronize();

            /**
             * @brief Destroy stream
             *
             * @todo what about pending operations?
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~Stream();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        /**
         * @brief Wrapper for CUdeviceptr reservations
         */
        class MemoryReservation {
        public:

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~MemoryReservation();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

        /**
         * @brief Wrapper for CUmemGenericAllocationHandle operations
         */
        class MemoryAllocation {
        public:

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~MemoryAllocation();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        };

    } // namespace cuda
} // namespace fractos::service::compute
