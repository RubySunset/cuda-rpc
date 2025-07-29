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
    class Library;
    class Kernel;
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


            // cuLibraryLoadData
            [[nodiscard]] core::future<std::shared_ptr<Library>>
            library_load_data(core::cap::memory& contents,
                              const std::vector<CUjit_option>& jit_options,
                              const std::vector<void*>& jit_values,
                              const std::vector<CUlibraryOption>& lib_options,
                              const std::vector<void*>& lib_values);
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

            // cudaGetDeviceProperties
            core::future<cudaDeviceProp> get_properties() const;

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
            [[nodiscard]] core::future<unsigned int>
            get_api_version();

            // cuCtxGetDevice
            std::shared_ptr<Device>
            get_device();

            // cuCtxGetLimit
            [[nodiscard]] core::future<size_t>
            get_limit(CUlimit limit);

            // cuCtxSynchronize
            [[nodiscard]] fractos::core::future<void>
            synchronize();


            // cuModuleLoad
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            module_load(const std::string path);

            // cuModuleLoadData
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            module_load_data(core::cap::memory& contents);


            // cuMemAlloc
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            mem_alloc(size_t size);

            // cuMemGetInfo (returns pair<free, total>)
            [[nodiscard]] core::future<std::pair<size_t, size_t>>
            mem_get_info() const;

            // cuMemsetD{8,16,32}
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, uint8_t val, size_t width);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, uint16_t val, size_t width);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, uint32_t val, size_t width);

            // cuMemsetD{8,16,32}Async
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, uint8_t val, size_t width, Stream& stream);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, uint16_t val, size_t width, Stream& stream);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, uint32_t val, size_t width, Stream& stream);

            // cuMemsetD2D{8,16,32}
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, size_t pitch, uint8_t val, size_t width, size_t height);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, size_t pitch, uint16_t val, size_t width, size_t height);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, size_t pitch, uint32_t val, size_t width, size_t height);

            // cuMemsetD2D{8,16,32}Async
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, size_t pitch, uint8_t val, size_t width, size_t height, Stream& stream);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, size_t pitch, uint16_t val, size_t width, size_t height, Stream& stream);
            [[nodiscard]] core::future<void>
            memset(CUdeviceptr addr, size_t pitch, uint32_t val, size_t width, size_t height, Stream& stream);


            // cuStreamCreate()
            [[nodiscard]] core::future<std::shared_ptr<Stream>>
            stream_create(CUstream_flags flags);


            // cuEventCreate()
            [[nodiscard]] core::future<std::shared_ptr<Event>>
            event_create(CUevent_flags flags);


            [[deprecated]]
            [[nodiscard]] core::future<std::shared_ptr<Module>>
            make_module_data(core::cap::memory& contents, uint64_t module_id);

            [[deprecated]]
            [[nodiscard]] core::future<std::shared_ptr<Memory>>
            make_memory(uint64_t size); // size_t make_memory(size_t size);

            [[deprecated]]
            [[nodiscard]] core::future<std::shared_ptr<Event>>
            make_event(fractos::wire::endian::uint32_t flags); // blocking or not

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
        };

        std::string to_string(const Context& obj);


        /** 
         *  @brief :Wrapper for CUmodule operations
         */
        class Module : public common::service::CltBase<Module> {
        public:
            CUmodule get_module() const;

            // cuModuleGetFunction
            [[nodiscard]] core::future<std::shared_ptr<Function>>
            get_function(const std::string& name);

            // cuModuleGetGlobal
            [[nodiscard]] core::future<CUdeviceptr>
            get_global(const std::string& file_name);
        };

        std::string to_string(const Module& obj);

        /**
         * @brief Wrapper for CUfunction operations
         */
        class Function : public common::service::CltBase<Function> {
        public:
            CUfunction get_function() const;

            // cuFuncSetAttribute
            [[nodiscard]] core::future<void>
            set_attribute(CUfunction_attribute attrib, int value);

            // cuLaunchKernel
            [[nodiscard]] core::future<void>
            launch(const void** args, dim3 gridDim, dim3 blockDim, size_t sharedMem,
                   std::optional<std::reference_wrapper<Stream>> stream);

            // cuOccupancyMaxActiveBlocksPerMultiprocessor
            [[nodiscard]] core::future<int>
            occupancy_max_active_blocks_per_multiprocessor_with_flags(
                int block_size, size_t dynamic_mem_size);

            // cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
            [[nodiscard]] core::future<int>
            occupancy_max_active_blocks_per_multiprocessor_with_flags(
                int block_size, size_t dynamic_mem_size, CUoccupancy_flags flags);

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


        // CUlibrary
        class Library : public common::service::CltBase<Library> {
        public:
            CUlibrary get_library() const;

            // cuLibraryGetKernel
            [[nodiscard]] core::future<std::shared_ptr<Kernel>>
            get_kernel(const std::string& name);
        };

        std::string to_string(const Library& obj);


        // CUkernel
        class Kernel : public common::service::CltBase<Kernel> {
        public:
            CUkernel get_kernel() const;

            // cuKernelGetFunction
            [[nodiscard]] core::future<std::shared_ptr<Function>>
            get_function(Context& ctx);
        };

        std::string to_string(const Kernel& obj);


        /**
         * @brief Wrapper for CUstream operations
         */
        class Stream : public common::service::CltBase<Stream> {
        public:
            // NOTE: result can be null
            std::shared_ptr<Context> get_context() const;

            CUstream get_stream() const;

            // cuStreamSynchronize
            [[nodiscard]] core::future<void>
            synchronize();

            // cuStreamWaitEvent
            [[nodiscard]] core::future<void>
            wait_event(Event& event, CUevent_wait_flags flags);
        };

        std::string to_string(const Stream& obj);

        /**
         * @brief Wrapper for CUevent operations
         */
        class Event : public common::service::CltBase<Event> {
        public:
            CUevent get_event() const;

            // cuEventSynchronize
            [[nodiscard]] core::future<void>
            synchronize();

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
        };

        std::string to_string(const Event& obj);


        /**
         *  @brief :Wrapper for CUdeviceptr reservations
         */
        class Memory : public common::service::CltBase<Memory> {
        public:
            CUdeviceptr get_deviceptr();
            fractos::core::cap::memory& get_cap_mem();
        };

        std::string to_string(const Memory& obj);

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
