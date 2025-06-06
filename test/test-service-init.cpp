#include <filesystem>
#include <fractos/common/cmdline.hpp>
#include <fractos/common/logging.hpp>
#include <fractos/common/signal.hpp>
#include <fractos/core/gns.hpp>
#include <fractos/service/compute/cuda.hpp>


using namespace fractos;

int
main(int argc, char *argv[])
{
    common::logging::init(argv[0]);

    auto odesc = common::cmdline::options();
    auto [args, ch] = common::cmdline::parse(odesc, argc, argv);
    common::signal::init_log_handler(SIGUSR1, ch->get_process());
    auto gns = core::gns::make_service();

    auto path = std::filesystem::path(argv[0]).parent_path();

    auto srv = service::compute::cuda::make_service(ch, *gns, "fractos::service::compute::cuda").get();
    srv->init(0).get();

    CHECK(srv->device_get_count().get() > 0);
    auto dev = srv->device_get(0).get();
    LOG(INFO) << "device 0: " << service::compute::cuda::to_string(*dev) << " - " << dev->get_name().get();
    CHECK(dev->total_mem().get() > 0);

    auto ctx = dev->make_context(0).get();

    {
        auto mod = ctx->module_load(path / "module.ptx").get();
        auto f0 = mod->get_function("f0").get();
        auto f1 = mod->get_function("f1").get();
        auto f2 = mod->get_function("f2").get();
        f0->destroy().get();
        f1->destroy().get();
        f2->destroy().get();
        mod->destroy().get();
    }

    {
        auto str = ctx->make_stream(CU_STREAM_DEFAULT, 0).get();
        str->synchronize().get();
        str->destroy().get();
    }

    ctx->destroy().get();
    dev->destroy().get();

    LOG(INFO) << "test done";
}
