#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <glog/logging.h>
#include <signal.h>
#include <utility>
#include <fstream>

#include <cuda.h>
#include <fractos/service/compute/cuda.hpp>
#include <nvtx3/nvToolsExt.h>
#include <cuda_profiler_api.h>


using namespace fractos;
using namespace fractos::service::compute::cuda;
using namespace std::chrono_literals;
using namespace fractos::core;

std::unique_ptr<fractos::service::compute::cuda::Service>
profile_experiment(std::unique_ptr<fractos::service::compute::cuda::Service> srv,
    std::shared_ptr<core::channel> ch, size_t N)
{
    using clock = std::chrono::high_resolution_clock;
    std::chrono::microseconds t_usec;

   
    
    ////////////////////////////////////////////////
    // 3) Create objects and manipulate their state

    LOG(INFO) << "==================start profile =============";

    LOG(INFO) << "start making vde";

    auto vdev = srv->make_device(0).get();

    LOG(INFO) << "start making ctx";
    auto vctx = vdev->make_context(1).get();

    LOG(INFO) << "start making stream ";
    auto stream = vctx->make_stream(CU_STREAM_DEFAULT, 0).get(); // 0 for default
    

    LOG(INFO) << "=============================================";

    size_t size = N * sizeof(int);


    auto mem_a = vctx->make_memory(size).get();
    auto mem_b = vctx->make_memory(size).get();
    auto mem_r = vctx->make_memory(size).get();

    auto mem_a_addr = mem_a->get_addr();
    auto mem_b_addr = mem_b->get_addr();
    auto mem_r_addr = mem_r->get_addr();  


    int* h_A = (int*)malloc(size);
    int* h_B = (int*)malloc(size);
    int* h_C = (int*)malloc(size);

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = static_cast<int>(i);
        h_B[i] = static_cast<int>(i * 2);

        h_C[i] = static_cast<int>(0);
    }


    auto mr_A = ch->make_memory_region(h_A, size, core::memory_region::translation_type::PIN);
    auto mr_B = ch->make_memory_region(h_B, size, core::memory_region::translation_type::PIN);
    auto mr_C = ch->make_memory_region(h_C, size, core::memory_region::translation_type::PIN);

    auto mem_a_local = ch->make_memory(h_A, size, *mr_A).get(); // ?
    auto mem_b_local = ch->make_memory(h_B, size, *mr_B).get();
    auto mem_r_local = ch->make_memory(h_C, size, *mr_C).get();

    ch->copy(mem_a_local, mem_a->get_cap_mem()).get(); 
    ch->copy(mem_b_local, mem_b->get_cap_mem()).get();   


    LOG(INFO) << "start module load";

    std::ifstream so_file ("/home/mingxuanyang/fractos/experiments/deps/app-compute-cuda/test.ptx", std::ios::in | std::ios::binary);
    size_t length;
    char* buffer;
    if (so_file) {

        LOG(INFO) << "find ptx kernel file";

    }
    so_file.seekg(0, so_file.end);
    length = static_cast<size_t>(so_file.tellg());
    so_file.seekg(0, so_file.beg);


    buffer = (char*)malloc(length);
    so_file.read(buffer, length);
    so_file.close();

    auto mem_so = ch->make_memory(buffer, length).get();


    
    LOG(INFO) << "Starting to register function";
    auto mod = vctx->make_module_data(mem_so, 2).get(); // data mem_so

    std::string func_name = "add";
    auto func = mod->get_function(func_name).get();


    std::array<size_t, 6> grid = {1024, 1, 1, 1024, 1, 1};
    func->call(*stream, grid, mem_a_addr, mem_b_addr, mem_r_addr, N).get(); // *stream
    stream->synchronize().get();
    vctx->synchronize().get();

    ch->copy(mem_r->get_cap_mem(), mem_r_local).get();

    // Verify the result
    bool resultIsCorrect = true;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            resultIsCorrect = false;
            std::cout << i << "  " << N << std::endl;
            break;
        }
    }

    if (resultIsCorrect) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }

    cuMemFreeHost((void*)h_A);
    cuMemFreeHost((void*)h_B);
    cuMemFreeHost((void*)h_C);


    LOG(INFO) << "destroy all mem";
    mem_a->destroy().get();
    mem_b->destroy().get();
    mem_r->destroy().get();


    LOG(INFO) << "destroy others";
    func->func_destroy().get();
    mod->destroy().get();
    stream->destroy().get();
    vctx->destroy().get();
    vdev->destroy().get();
    
    LOG(INFO) << "===================== DONE ==================";

    return srv;


}

int main(int argc, char *argv[])
{
    common::logging::init(argv[0]);
    LOG(INFO) <<"enter main client";

    //////////////////////////////////////////////////
    // 1) Parse command line

    /*
     * We let the client choose the remote service name, which should match the
     * one set when starting the service instance. This name is used to connect
     * the client to the service via GNS.
     */

    auto odesc = common::cmdline::options();
    odesc.add_options()
        ("service-name", common::cmdline::po::value<std::string>()->default_value("service"),
         "name used to publish example::service in GNS");
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);

    

    // Retrieve additional command line arguments
    auto name = args["service-name"].as<std::string>();

    // Log process state when receiving SIGUSR1
    common::signal::init_log_handler(SIGUSR1, ch->get_process());

    // Log current process state
    ch->get_process()->log_state();

    //////////////////////////////////////////////////
    // 2) Get service object, as registered by the server

    auto gns = core::gns::make_service();

    auto srv = fractos::service::compute::cuda::make_service(*gns, name, ch).get();
    LOG(INFO) << "================== start make_service========" ;

    
    size_t N = 1024;
    for(int i = 0; i < 2; i++) // 5
    {
        // // Move srv to a temporary unique_ptr
        // std::unique_ptr<fractos::service::compute::cuda::Service> temp_srv = std::move(srv);

        srv = profile_experiment(std::move(srv), ch, N);

        // Check if srv is still valid
        if (!srv) {
            std::cerr << "srv is null after move" << std::endl;
            break;
        }
    }

    LOG(INFO) << "================== end ========" ;

     
    return 0;

}   
