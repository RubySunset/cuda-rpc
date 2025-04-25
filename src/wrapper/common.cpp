#include <stdlib.h>

#include "./common.hpp"

std::string
get_env(std::string env_name, std::string default_str)
{
    auto res = default_str;
    auto env_val = secure_getenv(env_name.c_str());
    if (env_val) {
        res = env_val;
    }
    return res;
}
