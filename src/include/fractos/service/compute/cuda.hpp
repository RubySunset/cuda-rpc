#pragma once

#include <any>
#include <cuda.h>
#include <fractos/core/future.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/wire/endian.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <utility>




#define checkCudaErrors(err)  fractos::service::compute::cuda::ErrorChecker(err, __FILE__, __LINE__)

namespace fractos::service::compute { namespace [[gnu::visibility("default")]] cuda {

        class Service;
        class Device;
        class Context;
        class Module;
        class Function;
        class Stream;
        class Memory;
        class MemoryAllocation;
        class MemoryReservation;
        //event
        //graph
        //module
        

        struct no_Service_error : public std::runtime_error {
            no_Service_error(const std::string& what);
        };
    
    

        struct ErrorChecker {
            ErrorChecker(CUresult err);
            // ErrorChecker(CUresult err,  const char *file, const int line);

            ErrorChecker(CUresult err, const std::string& file, int line);

        private:
            void handleError(CUresult err, const std::string& file, int line);
            const CUresult err;
        };


        /**
         * @brief Connect to the CUDA Service with given name
         */
        [[nodiscard]] core::future<std::unique_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::gns::service& gns, const std::string& name,
                     const std::chrono::microseconds& wait_time = std::chrono::seconds{60});

        [[nodiscard]] core::future<std::unique_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::cap::request& Service_req);
        

        /**
         * @brief Connect to the CUDA Service with given name
         */             
        [[nodiscard]] core::future<std::unique_ptr<Service>>
        make_service(fractos::core::gns::service& gns, const std::string& name,
                    std::shared_ptr<core::channel> ch);
      
                     

        /**
         * The Service implicitly calls cuInit() when started.
         */
        class Service {
        public:
            std::shared_ptr<fractos::core::channel>
            get_default_channel();
        
            std::shared_ptr<fractos::core::channel>
            get_default_channel() const;
            
            void set_default_channel(std::shared_ptr<fractos::core::channel>
            ch);
            // [[nodiscard]] static fractos::core::future<std::unique_ptr<Service>>
            // make_service(fractos::core::gns::service& gns, const std::string& name,
            //                 std::shared_ptr<fractos::core::channel> ch);

            [[nodiscard]] fractos::core::future<std::shared_ptr<Device>> get_Device(
                fractos::core::gns::service& gns, uint8_t id);

            
            /**
             * @brief TODO: make_device with channel
             * @brief Wrapper for cuDeviceGet()
             */
            // [[nodiscard]] core::future<std::shared_ptr<Device>>
            // make_device(std::shared_ptr<core::channel> ch, uint64_t value);
            [[nodiscard]] core::future<std::shared_ptr<Device>>
            make_device(uint8_t value);

            /**
             * @brief Destroy Service connection, and all created objects
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            // Service();
            // ~Service();
            // // NOTE: not for public use
            Service(std::shared_ptr<void> pimpl);
            std::shared_ptr<void> _pimpl;
        };

        // std::string to_string(const Service& obj);

        /**
         * @brief Wrapper for CUdevice operations
         */
        class Device {
        public:
                                                                     
            /**
             * @brief TODO: transfer vector<CUctxCreateParams> through message
             * @brief Wrapper for cuCtxCreate_v4()
             */
            [[nodiscard]] core::future<std::shared_ptr<Context>>
            make_context(const std::vector<CUctxCreateParams>& paramsArray, unsigned int flags); // The paramsArray is an array of CUexecAffinityParam and the numParams describes the size of the array


            /**
             * @brief Wrapper for cuCtxCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<Context>>
            make_context(unsigned int flags);

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
           Device(std::shared_ptr<void> pimpl, fractos::wire::endian::uint8_t id);
           ~Device();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:

            bool _destroyed;
        
        };
        // std::string to_string(const Device& obj);

        /**
         * @brief Wrapper for CUcontext operations
         */
        class Context {
        public:
            /**
             * @brief Wrapper for cuModuleLoad()
             */
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            make_module_file(const std::string& file_name); // cubin PTX fatbin 

            // /**
            //  * @brief Wrapper for cuModuleLoad()
            //  */
            // [[nodiscard]] core::future<void>
            // make_module_file(const std::string& file_path); // cubin PTX fatbin 

            /**
             * @brief Wrapper for cuMemAlloc()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory(uint64_t size); // size_t make_memory(size_t size);

             /**
             * @brief TODO:Wrapper for cuModuleLoadData()
             */
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            make_module(const core::cap::memory& contents);

            /**
             * @brief TODO:Wrapper for cuStreamCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<Stream>>
            make_stream(CUstream_flags flags);

            /**
             * @brief TODO:Wrapper for cuMemAllocHost()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory_host(size_t size);

            /**
             * @brief TODO:Wrapper for cuMemAllocManaged()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory_managed(size_t size, CUmemAttach_flags flags);

            /**
             * @brief TODO:Wrapper for cuMemAddressReserve()
             */
            [[nodiscard]] core::future<std::shared_ptr<MemoryReservation>>
            make_memory_reservation(size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags);

            /**
             * @brief TODO:Wrapper for cuMemCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<MemoryAllocation>>
            make_memory_allocation(size_t size, const std::vector<CUmemAllocationProp>& props, unsigned long long flags);

            /**
             * @brief TODO:Wrapper for cuMemMap()
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
            Context(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t value);
            ~Context();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:
            bool _destroyed;
        
        };
        // std::string to_string(const Context& obj);

        


        /** 
         *  @brief :Wrapper for CUmodule operations
         */
        class Module {
        public:
                                                                  
            /**
             * @brief Wrapper for cuModuleGetFunction()
             */
            [[nodiscard]] core::future<std::shared_ptr<Function>>
            get_function(const std::string& file_name); // The paramsArray is an array of CUexecAffinityParam and the numParams describes the size of the array
            
            /**
             * @brief Destroy module with cuModuleUnload 
             */
            [[nodiscard]] core::future<void> 
            destroy();

        public:
            Module(std::shared_ptr<void> pimpl, std::string name);
            ~Module();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:
            bool _destroyed;
        
        };
        // std::string to_string(const Module& obj);

        /**
         * @brief Wrapper for CUfunction operations
         */
        class Function {
        public:
            /**
             * @brief Wrapper for cuLaunchKernel()
             */
            template<class... Args>
            [[nodiscard]]core::future<void> 
            call(std::pair<size_t, size_t>& gpu_grid, Args&&... ker_args);

             /**
             * @brief TODO: Wrapper for cuLaunchKernel()
             */
            template<class... Args>
            [[nodiscard]] core::future<void>
            call(Stream& stream, const std::tuple<size_t, size_t, size_t>& grid, Args&&... args);

            /**
             * @brief Destroy function 
             */
            [[nodiscard]] core::future<void>
            func_destroy();

        public:
            Function(std::shared_ptr<void> pimpl, std::string function_name);
            ~Function();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:
            bool _destroyed;
        
        };
        std::string to_string(const Function& obj);

        /**
         * @brief Wrapper for CU stream
         */
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
         *  @brief :Wrapper for CUdeviceptr reservations
         */
        class Memory {
        public:
            /**
             * @brief Destroy memory 
             */
            [[nodiscard]] core::future<void>
            destroy();


        public:
            Memory(std::shared_ptr<void> pimpl, fractos::wire::endian::uint64_t size);
            ~Memory();

            char* get_addr();
            fractos::core::cap::memory& get_cap_mem();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:

            bool _destroyed;
        
        };

        class MemoryAllocation : public Memory {
        public:
            /**
             * @brief Destroy memory 
             */
            [[nodiscard]] core::future<void>
            destroy();


        public:
            MemoryAllocation(std::shared_ptr<void> pimpl, fractos::wire::endian::uint64_t size);
            ~MemoryAllocation();

            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:
            bool _destroyed;
            
        };


        /**
         * @brief Wrapper for CUmemGenericAllocationHandle operations
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

    } // namespace cuda
} // namespace fractos::Service::compute
// #include "/home/mingxuanyang/fractos/experiments/deps/service-compute-cuda/src/library/functio.inc.hpp" // Include the implementation
#include "function.inc.hpp"