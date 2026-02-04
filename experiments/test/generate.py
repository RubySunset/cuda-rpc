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
            'komodo',
        ],
        test=[
            "test-service-init",
            "test-service-memory",
            "test-stream",
            "test-memcpy",
            "test-cublas-wrapper-simple",
            "test-cublas-wrapper-full",
        ],
    )
    e.params(env="""
      env:
        LD_LIBRARY_PATH: ${DIST}/lib:${DIST}/lib/libfractos-service-compute-cuda-wrapper\
""")

    # copy files
    for elem in copy:
        src, dst = elem.split("=")
        e.pack(src, dst)

    # generate files

    e.params(
        ID="{{test}}",
        DONE="res/{{ID}}",
        FAIL="logs/{{ID}}.fail",
        CMD="exec python3 -u ./run.py run env/exp-{{ID}}.yaml",
    )

    e.params(include=make_include_loader())
    e.generate(str(src_dir / "exp.yaml"), "env/exp-{{ID}}.yaml")

    e.generate_jobs("shell","jobs/{{ID}}.sh",
                    # variables available to FILTER
                    export=[
                        "ID",
                        "test",
                        "cluster",
                    ],
                    depends=[
                        "env/exp-{{ID}}.yaml",
                        "dist-amd64/bin/{{test}}",
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
