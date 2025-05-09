#include "common.hpp"

bool
impl::Base::is_destroyed() const
{
    return _destroyed.test();
}

bool
impl::Base::destroy_maybe()
{
    auto was_destroyed = _destroyed.test_and_set();
    return not was_destroyed;
}
