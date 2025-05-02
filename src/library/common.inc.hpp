#include "fractos/core/future.hpp"
#include <common.hpp>


template <class Tsrv, class Timpl>
inline
Timpl &
impl::Base<Tsrv, Timpl>::get(Tsrv& obj)
{
    return *reinterpret_cast<Timpl*>(obj._pimpl.get());
}

template <class Tsrv, class Timpl>
inline
const Timpl &
impl::Base<Tsrv, Timpl>::get(const Tsrv& obj)
{
    return *reinterpret_cast<const Timpl*>(obj._pimpl.get());
}

template <class Tsrv, class Timpl>
inline
fractos::core::future<void>
impl::Base<Tsrv, Timpl>::destroy()
{
    if (not _destroyed.test_and_set()) {
        return do_destroy();
    } else {
        return fractos::core::make_ready_future();
    }
}
