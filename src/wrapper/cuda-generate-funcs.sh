#!/bin/bash -e

NAME=$1
LIBCUDA=$(readlink -f $2)
OUTPUT=$3

generate() {
    name=$1
    libcuda=$2
    output=$3

    syms=$(readelf -Ws "$libcuda" | \
               grep -v " UND " | grep " FUNC " | grep " DEFAULT ")

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
cuda_function_t ${1}_default_functions[];

[[gnu::visibility("hidden")]]
cuda_function_t ${1}_default_functions[] = {
EOF

    for sym in $syms; do
        echo "    {\"$sym\", (void*)$sym}," >>"$output"
    done

    echo "    {nullptr, nullptr}," >>"$output"
    echo "};" >>"$output"
}

generate "$NAME" "$LIBCUDA" "$OUTPUT"
