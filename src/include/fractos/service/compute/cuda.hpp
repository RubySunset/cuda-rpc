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
        class Event;
        class Memory;
        class MemoryAllocation;
        class MemoryReservation;
        //graph

        

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
         * @brief Connect to the CUDA Service
         */
        [[nodiscard]] core::future<std::unique_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::gns::service& gns, const std::string& name,
                     const std::chrono::microseconds& wait_time = std::chrono::seconds{60});

        /**
         * @brief Connect to the CUDA Service
         */
        [[nodiscard]] core::future<std::unique_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     const core::cap::request& connect);

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

            const core::cap::request& get_connect() const;


            // cuDriverGetversion
            [[nodiscard]] fractos::core::future<int>
            get_driver_version();


            // cuInit
            [[nodiscard]] fractos::core::future<void>
            init(unsigned int flags);


            // cuDeviceGet
            [[nodiscard]] fractos::core::future<std::shared_ptr<Device>>
            device_get(int ordinal);

            // cuDeviceGetCount
            [[nodiscard]] fractos::core::future<int>
            device_get_count();


            // cuModuleGetLoadingMode
            [[nodiscard]] fractos::core::future<CUmoduleLoadingMode>
            module_get_loading_mode();


            /**
             * @brief Destroy Service connection, and all created objects
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            // NOTE: not for public use
            Service(std::shared_ptr<void> pimpl);
            std::shared_ptr<void> _pimpl;
        };

        std::string to_string(const Service& obj);

        /**
         * @brief Wrapper for CUdevice operations
         */
        class Device {
        public:

            CUdevice get_device() const;

            // cuDeviceGetAttribute
            core::future<int> get_attribute(CUdevice_attribute) const;

            // cuDeviceGetName
            core::future<std::string> get_name() const;

            // cuDeviceGetUuid
            core::future<CUuuid> get_uuid() const;

            // cuDeviceTotalMem
            core::future<size_t> total_mem() const;

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
            // NOTE: not for public use
            Device(std::shared_ptr<void> pimpl);
            ~Device();
            std::shared_ptr<void> _pimpl;
        private:

            bool _destroyed;
        
        };

        std::string to_string(const Device& obj);

        /**
         * @brief Wrapper for CUcontext operations
         */
        class Context {
        public:
            CUcontext get_context() const;

            // cuCtxGetDevice
            std::shared_ptr<Device> get_device();

            // /**
            //  * @brief Wrapper for cuModuleLoad() - not in use
            //  */
            // [[nodiscard]] core::future<std::shared_ptr<Module>>
            // make_module_file(const std::string& file_path); // cubin PTX fatbin 

            /**
             * @brief TODO:Wrapper for cuModuleLoadData()
             */
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            make_module_data(core::cap::memory& contents, uint64_t module_id);

            /**
             * @brief Wrapper for cuMemAlloc()
             */
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory(uint64_t size); // size_t make_memory(size_t size);

            [[nodiscard]] core::future<void> 
            make_memory_rpc_test(uint64_t size); // size_t make_memory(size_t size);

            /**
             * @brief TODO:Wrapper for cuEventCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<Event>>
            make_event(fractos::wire::endian::uint32_t flags); // blocking or not

            /**
             * @brief TODO:Wrapper for cuStreamCreate()
             */
            [[nodiscard]] core::future<std::shared_ptr<Stream>>
            make_stream(CUstream_flags flags, fractos::wire::endian::uint32_t id); // blocking or not

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
            Module(std::shared_ptr<void> pimpl, uint64_t module_id);
            Module(std::shared_ptr<void> pimpl, core::cap::memory contents, std::string name);
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

            template<class... Args>
            [[nodiscard]]core::future<void> 
            call(std::array<size_t, 6>& gpu_grid, Args&&... ker_args);

            template<class... Args>
            [[nodiscard]]core::future<void> 
            call(Stream& stream, std::pair<size_t, size_t>& gpu_grid, Args&&... ker_args);

            template<class... Args>
            [[nodiscard]]core::future<void> 
            call(Stream& stream,std::array<size_t, 6>& gpu_grid, Args&&... ker_args);

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
        // std::string to_string(const Function& obj);

        /**
         * @brief Wrapper for CUstream operations
         */
        class Stream{
        public:
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
            Stream(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t flags, fractos::wire::endian::uint32_t id);
            ~Stream();
            fractos::wire::endian::uint32_t get_stream_id();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
            
        private:
            bool _destroyed;
            fractos::wire::endian::uint32_t _id;
        };

        /**
         * @brief Wrapper for CUevent operations
         */
        class Event{
        public:
            /**
             * @brief Wrapper for cuEventElapsedTime()
             */
            [[nodiscard]] fractos::core::future<fractos::wire::endian::uint32_t> // memcpy uint32 to float in IEEE 754 format
            make_event_time(Event& evt1, Event& evt2);


            /**
             * @brief Wrapper for cuEventRecord()
             */
            [[nodiscard]] fractos::core::future<void>
            make_record(Stream& stream);


            /**
             * @brief Wrapper for cuEventSynchronize()
             */
            [[nodiscard]] fractos::core::future<void>
            synchronize();

            /**
             * @brief Wrapper for cuEventDestroy()
             *
             * @todo what about pending operations?
             */
            [[nodiscard]] core::future<void>
            destroy();

        public:
            Event(std::shared_ptr<void> pimpl, fractos::wire::endian::uint32_t flags);
            ~Event();
            fractos::wire::endian::uint32_t get_event_id();
            // NOTE: not for public use
            std::shared_ptr<void> _pimpl;
            
        private:
            bool _destroyed;
            fractos::wire::endian::uint32_t _id;
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
} // namespace fractos::service::compute
