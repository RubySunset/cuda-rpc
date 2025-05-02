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
