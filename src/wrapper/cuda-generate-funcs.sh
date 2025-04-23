#!/bin/bash -e

LIBCUDA=$(readlink -f $1)
OUTPUT=$2

generate() {
    libcuda=$1
    output=$2

    syms=$(readelf -Ws "$libcuda" | grep -v " UND " | grep " FUNC " | grep " DEFAULT " | tr -s " " | cut -f 9 -d " ")

    cat >"$output" <<EOF
#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#define xstr(x) #x
EOF

    for sym in $syms; do
        cat >>"$output" <<EOF
extern "C" [[gnu::weak, gnu::visibility("default")]] void $sym()
{
    LOG(FATAL) << "symbol not implemented: " xstr($sym);
}
EOF
    done

    cat >>"$output" <<EOF
struct cuda_function_t {
    char const* name;
    void* ptr;
};

extern "C" [[gnu::visibility("hidden")]]
cuda_function_t default_functions[];

[[gnu::visibility("hidden")]]
cuda_function_t default_functions[] = {
EOF

    for sym in $syms; do
        echo "    {\"$sym\", (void*)$sym}," >>"$output"
    done

    echo "    {nullptr, nullptr}," >>"$output"
    echo "};" >>"$output"
}

generate "$LIBCUDA" "$OUTPUT"
