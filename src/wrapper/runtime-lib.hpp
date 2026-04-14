#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <fatbinary_section.h>
#include <fractos/common/logging.hpp>
#include <fractos/logging.hpp>
#include <glog/logging.h>
#include <string>
#include <unordered_map>

#include <common.hpp>
#include "./runtime-state.hpp"
#include "./runtime-syms-extern.hpp"


struct RuntimeLibSyms {
#define SYM(name) decltype(&name) ptr_ ## name;
#include "./runtime-syms.hpp"
#undef SYM
};

RuntimeLibSyms & get_runtime_lib_syms();
