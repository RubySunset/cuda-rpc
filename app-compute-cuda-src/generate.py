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
        cluster=[
            'localhost',
#            'quokka',
        ],
        benchmark=[
            'simple',
            # 'rpc'
            # 'pipeline',
        ],
    )

    # copy files
    for elem in copy:
        src, dst = elem.split("=")
        e.pack(src, dst)

    # benchmark-specific parameters

    with e.view("benchmark == 'simple'") as ee:
        ee.params(
            # no workload-specific parameters
            workload="_",
            exp_template="exp-{{benchmark}}.yaml",
            binary="{{benchmark}}",
            metric=[
                "",
            ],
        )
    with e.view("benchmark == 'rpc'") as ee:
        ee.params(
            # no workload-specific parameters
            mode=[
                "manual",
            ],
            metric=[
                "latency",
            ],
            placement=[
                "same_node",
            ],
            binary="{{benchmark}}_{{mode}}",
            exp_template="exp-{{benchmark}}.yaml",
            workload="_",
        )
        # with ee.view("metric == 'latency'") as eee:
        #     eee.params(
        #         ctl_threads=["all_rr"],
        #         workload="mode={{mode}}-placement={{placement}}-ctl_threads={{ctl_threads}}",
        #         workload_cmd="{{cmd_latency}},target_clock_overhead_perc=1,target_stddev_perc=5,timed_batch_size_warmup=10,timed_batch_size_hint=10",
        #     )

    # with e.view("benchmark == 'pipeline'") as ee:
    #     ee.params(
    #         client_mode=["naive", "optimized"],
    #         # 0: service uses node0's controller (same as pipeline app)
    #         # 1: service uses node1's controller (different from pipeline app)
    #         service_ctl=[0, 1],
    #         workload="{{client_mode}}_{{service_ctl}}",
    #     )

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
