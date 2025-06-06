#pragma once

#include <fractos/common/cmdline.hpp>
#include <fractos/common/cpu.hpp>
#include <fractos/common/experiment.hpp>


static inline
auto
options()
{
    using namespace fractos::common;

    auto odesc = cmdline::options();
    odesc.add_options()
        ("metric", cmdline::po::value<std::string>()->required(),
         "target metric (see fractos::common::experiment::parse_metric)")
        ("output", cmdline::po::value<std::string>(),
         "output file")
        ("control-thread", cmdline::po::value<std::string>()->value_name("CPUSET")->default_value("+all+1"),
         "CPU set to pin the control thread to (see fractos::common::cpu::parse_set())")
        ("measurement-threads", cmdline::po::value<std::string>()->value_name("CPUSET")->default_value("+all"),
         "CPU set to create and pin measurement threads (see fractos::common::cpu::parse_set())")
        ;
    return odesc;
}

static inline auto
parse(auto& odesc, auto argc, auto argv)
{
    using namespace fractos::common;

    auto [args, pch] = cmdline::parse(odesc, argc, argv);

    auto metric = experiment::parse_metric(args["metric"].template as<std::string>());
    // if (metric.get_name() == "latency" and GPROF || PERF) {
    //     LOG(INFO) << "setting fixed-length experiment";
    //     metric.params.latency.stddev_perc = 0;
    //     params.confidence_sigma = 0;
    //     params.batch_group_size = 1;
    //     params.batch_size_warmup = 0;
    //     params.batch_size = 20000000;
    // }

    std::string output = "";
    if (args.count("output")) {
        output = args["output"].template as<std::string>();
    }

    auto base = cpu::get_current_set();

    auto control_thread = cpu::parse_set(*base, args["control-thread"].template as<std::string>());
    if (control_thread->size() == 0) {
        std::cerr << "Error: empty cpuset for --control-thread" << std::endl;
        exit(1);
    }
    LOG(INFO) << "Control thread: " << cpu::to_string(*control_thread);

    auto measurement_threads = cpu::parse_set(*base, args["measurement-threads"].template as<std::string>());
    if (measurement_threads->size() == 0) {
        std::cerr << "Error: empty cpuset for --measurement-threads" << std::endl;
        exit(1);
    }
    LOG(INFO) << "Measurement threads: " << cpu::to_string(*measurement_threads);

    return std::make_tuple(args, pch, output, metric, control_thread, measurement_threads);
}



static inline
auto
options_single_thread()
{
    using namespace fractos::common;

    auto odesc = cmdline::options();
    odesc.add_options()
        ("output", cmdline::po::value<std::string>(),
         "output file")
        ("control-thread", cmdline::po::value<std::string>()->value_name("CPUSET")->default_value("+0"),
         "CPU set to pin this program to (see fractos::common::cpu::parse_set())")
        ;
    return odesc;
}

static inline auto
parse_single_thread(auto& odesc, auto argc, auto argv)
{
    using namespace fractos::common;

    auto [args, pch] = cmdline::parse(odesc, argc, argv);

    auto output = args["output"].template as<std::string>();

    auto base = cpu::get_current_set();

    auto control_thread = cpu::parse_set(*base, args["control-thread"].template as<std::string>());
    if (control_thread->size() == 0) {
        std::cerr << "Error: empty cpuset for --control-thread" << std::endl;
        exit(1);
    }
    LOG(INFO) << "Control thread: " << cpu::to_string(*control_thread);

    return std::make_tuple(args, pch, output, control_thread);
}


static inline
auto
options_multi_thread()
{
    using namespace fractos::common;

    auto odesc = cmdline::options();
    odesc.add_options()
        ("output", cmdline::po::value<std::string>(),
         "output file")
        ("control-thread", cmdline::po::value<std::string>()->value_name("CPUSET")->default_value("-0"),
         "CPU set to pin the control thread to (see fractos::common::cpu::parse_set())")
        ("measurement-threads", cmdline::po::value<std::string>()->value_name("CPUSET")->default_value("+all"),
         "CPU set to create and pin measurement threads (see fractos::common::cpu::parse_set())")
        ;
    return odesc;
}

static inline auto
parse_multi_thread(auto& odesc, auto argc, auto argv)
{
    using namespace fractos::common;

    auto [args, pch] = cmdline::parse(odesc, argc, argv);

    std::string output = "";
    if (args.count("output")) {
        output = args["output"].template as<std::string>();
    }

    auto base = cpu::get_current_set();

    auto control_thread = cpu::parse_set(*base, args["control-thread"].template as<std::string>());
    if (control_thread->size() == 0) {
        std::cerr << "Error: empty cpuset for --control-thread" << std::endl;
        exit(1);
    }
    LOG(INFO) << "Control thread: " << cpu::to_string(*control_thread);

    auto measurement_threads = cpu::parse_set(*base, args["measurement-threads"].template as<std::string>());
    if (measurement_threads->size() == 0) {
        std::cerr << "Error: empty cpuset for --measurement-threads" << std::endl;
        exit(1);
    }
    LOG(INFO) << "Measurement threads: " << cpu::to_string(*measurement_threads);

    return std::make_tuple(args, pch, output, control_thread, measurement_threads);
}
