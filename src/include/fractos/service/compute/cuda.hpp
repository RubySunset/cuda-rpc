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


#define checkCudaErrors(err)  fractos::service::compute::cuda::ErrorChecker(err, __FILE__, __LINE__)

namespace fractos::service::compute { namespace [[gnu::visibility("default")]] cuda {

        class cuda_service;
        class cuda_device;
        class cuda_context;
        class cuda_memory;
        class cuda_module;
        class cuda_function;
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
        

        struct no_cuda_service_error : public std::runtime_error {
            no_cuda_service_error(const std::string& what);
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
         * @brief Connect to the CUDA cuda_service with given name
         */
        [[nodiscard]] core::future<std::unique_ptr<cuda_service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::gns::service& gns, const std::string& name,
                     const std::chrono::microseconds& wait_time = std::chrono::seconds{60});

        [[nodiscard]] core::future<std::unique_ptr<cuda_service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::cap::request& cuda_service_req);
        

        /**
         * @brief Connect to the CUDA cuda_service with given name
         */             
        [[nodiscard]] core::future<std::unique_ptr<cuda_service>>
        make_cuda_service(fractos::core::gns::service& gns, const std::string& name,
                    std::shared_ptr<core::channel> ch);
      
                     

        /**
         * The cuda_service implicitly calls cuInit() when started.
         */
        class cuda_service {
        public:
            std::shared_ptr<fractos::core::channel>
            get_default_channel();
        
            std::shared_ptr<fractos::core::channel>
            get_default_channel() const;
            
            void set_default_channel(std::shared_ptr<fractos::core::channel>
            ch);
            // [[nodiscard]] static fractos::core::future<std::unique_ptr<cuda_service>>
            // make_cuda_service(fractos::core::gns::service& gns, const std::string& name,
            //                 std::shared_ptr<fractos::core::channel> ch);

            [[nodiscard]] fractos::core::future<std::shared_ptr<cuda_device>> get_cuda_device(
                fractos::core::gns::service& gns, uint8_t id);

            
            /**
             * @brief Wrapper for cuDeviceGet()
             */
            // [[nodiscard]] core::future<std::shared_ptr<cuda_device>>
            // make_cuda_device(std::shared_ptr<core::channel> ch, uint64_t value);

            [[nodiscard]] core::future<std::shared_ptr<cuda_device>>
            make_cuda_device(uint8_t value);

            /**
             * @brief Destroy cuda_service connection, and all created objects
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            // cuda_service();
            // ~cuda_service();
            // // NOTE: not for public use
            cuda_service(std::shared_ptr<void> pimpl);
            cuda_service(std::string name);
            std::shared_ptr<void> _pimpl;
        };

        std::string to_string(const cuda_service& obj);

        /**
         * @brief Wrapper for CUdevice operations
         */
        class cuda_device {
        public:
        
            // fractos::core::future<void> destroy();
            cuda_device(std::shared_ptr<void> pimpl, fractos::wire::endian::uint8_t id);
            cuda_device(std::shared_ptr<void> pimpl);
        
            cuda_device(fractos::wire::endian::uint8_t id);  

                                                                               
            /**
             * @brief TODO: transfer vector<CUctxCreateParams> through message
             * @brief Wrapper for cuCtxCreate_v4()
             */
            [[nodiscard]] core::future<std::shared_ptr<Context>>
            make_cuda_context(const std::vector<CUctxCreateParams>& paramsArray, unsigned int flags); // The paramsArray is an array of CUexecAffinityParam and the numParams describes the size of the array


            /**
             * @brief Wrapper for cuCtxCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<cuda_context>>
            make_cuda_context(unsigned int flags);

            /**
             * @brief Destroy device and all its contents
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            ~cuda_device();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:

            bool _destroyed;
        
        };
        std::string to_string(const cuda_device& obj);

        /**
         * @brief Wrapper for CUcontext operations
         */
        class cuda_context {
        public:
        
            // fractos::core::future<void> destroy();
            cuda_context(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t value);
            cuda_context(std::shared_ptr<void> pimpl);
        
            cuda_context(fractos::wire::endian::uint32_t value);

            /**
             * @brief Wrapper for cuModuleLoad()
             */
            [[nodiscard]] core::future<std::shared_ptr<cuda_module>>
            make_module_file(const std::string& file_name); // cubin PTX fatbin 

            // /**
            //  * @brief Wrapper for cuModuleLoad()
            //  */
            // [[nodiscard]] core::future<void>
            // make_module_file(const std::string& file_path); // cubin PTX fatbin 

            /**
             * @brief Wrapper for cuModuleLoadData()
             */
            [[nodiscard]] core::future<std::shared_ptr<cuda_module>>
            make_module_data(const core::cap::memory& contents);

            /**
             * @brief Wrapper for cuMemAlloc()
             */
            [[nodiscard]] core::future<std::shared_ptr<cuda_memory>>
            make_cumemalloc(uint64_t size); // size_t

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
            ~cuda_context();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:

            bool _destroyed;
        
        };
        std::string to_string(const cuda_context& obj);


        class cuda_memory {
        public:
        
            // fractos::core::future<void> destroy();
            cuda_memory(std::shared_ptr<void> pimpl, fractos::wire::endian::uint64_t size);
            cuda_memory(std::shared_ptr<void> pimpl);
        
            cuda_memory(fractos::wire::endian::uint64_t size);

            

            /**
             * @brief Destroy memory 
             */
            [[nodiscard]] core::future<void>
            destroy();


        public:
            ~cuda_memory();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:

            bool _destroyed;
        
        };
        std::string to_string(const cuda_memory& obj);

        class cuda_module {
        public:

            cuda_module(std::shared_ptr<void> pimpl, std::string name);
            cuda_module(std::shared_ptr<void> pimpl);
        
            cuda_module(std::string name);  

                                                                                
            /**
             * @brief Wrapper for cuModuleGetFunction()
             */
            [[nodiscard]] core::future<std::shared_ptr<cuda_function>>
            get_cuda_function(const std::string& file_name); // The paramsArray is an array of CUexecAffinityParam and the numParams describes the size of the array
            
            /**
             * @brief Destroy module with cuModuleUnload 
             */
            [[nodiscard]] core::future<void> 
            destroy();

        public:
            ~cuda_module();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
        private:

            bool _destroyed;
        
        };
        std::string to_string(const cuda_module& obj);

        class cuda_function {
            public:
            
                // fractos::core::future<void> destroy();
                cuda_function(std::shared_ptr<void> pimpl, std::string function_name);
                cuda_function(std::shared_ptr<void> pimpl);
            
                cuda_function(std::string function_name);

                /**
                 * @brief Wrapper for cuLaunchKernel()
                 */
                template<class... Args>
                [[nodiscard]] core::future<void>
                call(std::pair<size_t, size_t>& gpu_grid, Args&&... args);

                /**
                 * @brief Wrapper for cuLaunchKernel()
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
                ~cuda_function();
                // NOTE: not for public use
                std::shared_ptr<void> _pimpl;
            private:
    
                bool _destroyed;
            
            };
            std::string to_string(const cuda_function& obj);
        

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
             * @brief Wrapper for cuModuleGetFunction()
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
} // namespace fractos::cuda_service::compute
