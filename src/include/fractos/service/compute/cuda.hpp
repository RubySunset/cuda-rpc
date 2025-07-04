#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <fractos/common/service/clt_base.hpp>
#include <fractos/core/future.hpp>
#include <fractos/core/channel.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/wire/endian.hpp>
#include <memory>
#include <string>
#include <vector>
#include <utility>



namespace fractos::service::compute { namespace [[gnu::visibility("default")]] cuda {
    class Service;
    class Device;
    class Context;
    class Module;
    class Function;
    class Stream;
    class Event;
    class Memory;
    class MemoryReservation;
    class MemoryAllocation;
} }

namespace fractos::service::compute { namespace [[gnu::visibility("default")]] cuda {

        struct CudaError : public std::runtime_error {
            CudaError(CUresult cuerror);
            const CUresult cuerror;
        };


        /**
         * @brief Connect to the CUDA Service
         */
        [[nodiscard]] core::future<std::shared_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     core::gns::service& gns, const std::string& name,
                     const std::chrono::microseconds& wait_time = std::chrono::seconds{60});

        /**
         * @brief Connect to the CUDA Service
         */
        [[nodiscard]] core::future<std::shared_ptr<Service>>
        make_service(std::shared_ptr<core::channel> ch,
                     const core::cap::request& connect);

        /**
         * The Service implicitly calls cuInit() when started.
         */
        class Service : public common::service::CltBase<Service> {
        public:
            /**
             * @brief Return a capability that can be used with make_service()
             */
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
        };

        std::string to_string(const Service& obj);

        /**
         * @brief Wrapper for CUdevice operations
         */
        class Device : public common::service::CltBase<Device> {
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
        };

        std::string to_string(const Device& obj);

        /**
         * @brief Wrapper for CUcontext operations
         */
        class Context : public common::service::CltBase<Context> {
        public:
            CUcontext get_context() const;

            // cuCtxGetApiVersion
            core::future<unsigned int> get_api_version();

            // cuCtxGetDevice
            std::shared_ptr<Device> get_device();

            // cuCtxGetLimit
            core::future<size_t> get_limit(CUlimit limit);

            // cuCtxSynchronize
            [[nodiscard]] fractos::core::future<void>
            synchronize();

            // cuMemAlloc
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            mem_alloc(size_t size);

            // cuModuleLoad
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            module_load(const std::string path);

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
            [[deprecated]]
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory(MemoryReservation& reservation, MemoryAllocation& allocation, unsigned long long flags);
        };

        std::string to_string(const Context& obj);

        


        /** 
         *  @brief :Wrapper for CUmodule operations
         */
        class Module : public common::service::CltBase<Module> {
        public:

            // cuModuleGetGlobal
            [[nodiscard]] core::future<CUdeviceptr>
            get_global(const std::string& file_name);


            /**
             * @brief Wrapper for cuModuleGetFunction()
             */
            [[nodiscard]] core::future<std::shared_ptr<Function>>
            get_function(const std::string& file_name); // The paramsArray is an array of CUexecAffinityParam and the numParams describes the size of the array
        };

        std::string to_string(const Module& obj);

        /**
         * @brief Wrapper for CUfunction operations
         */
        class Function : public common::service::CltBase<Function> {
        public:
            // cuLaunchKernel
            [[nodiscard]] core::future<void>
            launch(const void** args, dim3 gridDim, dim3 blockDim, size_t sharedMem,
                   std::optional<std::reference_wrapper<Stream>> stream);

            // shorthands with variadic kernel arguments

            template<class... Args>
            [[nodiscard]] core::future<void>
            launch(dim3 gridDim, dim3 blockDim, Args&&... args);

            template<class... Args>
            [[nodiscard]] core::future<void>
            launch(size_t sharedMem, dim3 gridDim, dim3 blockDim, Args&&... args);

            template<class... Args>
            [[nodiscard]] core::future<void>
            launch(Stream& stream, dim3 gridDim, dim3 blockDim, Args&&... args);

            template<class... Args>
            [[nodiscard]] core::future<void>
            launch(Stream& stream, size_t sharedMem, dim3 gridDim, dim3 blockDim, Args&&... args);

        public:
            void _launch_check_args(const std::vector<size_t>& args_size);
        };

        std::string to_string(const Function& obj);

        /**
         * @brief Wrapper for CUstream operations
         */
        class Stream : public common::service::CltBase<Stream> {
        public:
            /**
             * @brief Wrapper for cuStreamSynchronize()
             */
            [[nodiscard]] fractos::core::future<void>
            synchronize();

            // TODO: use cuda types
            fractos::wire::endian::uint32_t get_stream_id();
        };

        std::string to_string(const Stream& obj);

        /**
         * @brief Wrapper for CUevent operations
         */
        class Event : public common::service::CltBase<Event> {
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
        };

        std::string to_string(const Event& obj);


        /**
         *  @brief :Wrapper for CUdeviceptr reservations
         */
        class Memory : public common::service::CltBase<Memory> {
        public:
            char* get_addr();
            fractos::core::cap::memory& get_cap_mem();
        };

        std::string to_string(const Module& obj);

        class MemoryAllocation : public common::service::CltBase<MemoryAllocation> {
        };


        /**
         * @brief Wrapper for CUmemGenericAllocationHandle operations
         */
        class MemoryReservation : public common::service::CltBase<MemoryReservation> {
        };

    } // namespace cuda
} // namespace fractos::service::compute

#include <fractos/service/compute/cuda.inc.hpp>
