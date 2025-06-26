#!/usr/bin/env python3
# -*- python -*-

import argparse
from generate_common import *
import os
from pathlib import Path
import re
from sciexp2.expdef.env import *
import sys


def run(base_dir, output_dir, src_dir, copy):
    e = Experiments(out=str(output_dir))

    # base parameters
    e.params(
        "mode != 'cuda' or placement == 'local'",
        cluster=[
            'kea_and_komodo',
        ],
        benchmark=[
            'memory_alloc',
            'kernel_launch',
        ],
        mode=[
            "cuda",
            "wrapper",
            "fractos",
        ],
        placement=[
            "local",
            "remote",
        ],
        cmd_latency="--measurement-threads {{measurement_threads_value}} --metric latency,target_clock_overhead_perc=1,target_stddev_perc=5,measurements_hint=10,timed_batch_size_warmup=5,timed_batch_size_hint=1",
        cmd_latency_fixed="--measurement-threads {{measurement_threads_value}} --metric fixed_latency,warmup=100,total=100000,batch=20",
        # unused variables
        size="",
    )
    with e.view("placement == 'local'") as ee:
        ee.params(node="gpu")
    with e.view("placement == 'remote'") as ee:
        ee.params(node="cpu")
    with e.view("mode != 'wrapper'") as ee:
        ee.params(env="")
    with e.view("mode == 'wrapper'") as ee:
        ee.params(env="""
      env:
        LD_LIBRARY_PATH: ${DIST}/lib:${DIST}/lib/libfractos-service-compute-cuda-wrapper\
""")

    # copy files
    for elem in copy:
        src, dst = elem.split("=")
        e.pack(src, dst)

    # benchmark-specific parameters

    with e.view("benchmark == 'memory_alloc'") as ee:
        ee.params(
            size=[2**0, 2**10, 2**20, 2**22],
            # no workload-specific parameters
            workload="size={{size}}-mode={{mode}}-placement={{placement}}",
            exp_template="exp-simple.yaml",
            binary="app-{{benchmark}}",
            metric=[
                "latency",
            ],
        )
        with ee.view("metric == 'latency'") as eee:
            eee.params(measurement_threads_value="+all+1",
                       workload_cmd="{{cmd_latency_fixed}} --buffer-size {{size}} --mode {{mode}}")

    with e.view("benchmark == 'kernel_launch'") as ee:
        ee.params(
            # no workload-specific parameters
            workload="mode={{mode}}-placement={{placement}}",
            exp_template="exp-simple.yaml",
            binary="app-{{benchmark}}",
            metric=[
                "latency",
            ],
        )
        with ee.view("metric == 'latency'") as eee:
            eee.params(measurement_threads_value="+all+1",
                       workload_cmd="{{cmd_latency_fixed}} --ptx ${DIST}/bin/empty_kernel.ptx --mode {{mode}}")

    # generate files

    e.params(
        ID="{{cluster}}-{{benchmark}}-{{metric}}-{{workload}}",
        DONE="res/{{ID}}",
        FAIL="logs/{{ID}}.fail",
        CMD="exec python3 -u ./run.py run env/exp-{{ID}}.yaml",
    )

    e.params(include=make_include_loader())
    e.generate(str(src_dir / "{{exp_template}}"), "env/exp-{{ID}}.yaml")

    e.generate_jobs("shell","jobs/{{ID}}.sh",
                    # variables available to FILTER
                    export=[
                        "ID",
                        "benchmark",
                        "cluster",
                        "workload",
                        "mode",
                        "placement",
                        "size",
                        # "rps",
                    ],
                    depends=[
                        "env/exp-{{ID}}.yaml",
                        "dist-amd64/bin/{{binary}}",
                        "dist-amd64/bin/fractos-service-compute-cuda",
                    ])


def main(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("base_dir")
    parser.add_argument("output_dir")
    parser.add_argument("src_dir")
    parser.add_argument("--copy", action="append")

    args = parser.parse_args(args)
    run(
        base_dir=Path(args.base_dir),
        output_dir=Path(args.output_dir),
        src_dir=Path(args.src_dir),
        copy=args.copy,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
