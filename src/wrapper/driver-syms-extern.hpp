#pragma once

#define SYM(name) extern decltype(&name) ptr_ ## name;
#include "./driver-syms.hpp"
#undef SYM
