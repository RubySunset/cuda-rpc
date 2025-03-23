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
