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
    return destroy_maybe()
        .then([](auto& fut) {
            auto did_destroy = fut.get();
            if (not did_destroy) {
                throw std::runtime_error("cannot destroy object twice");
            }
        });
}

template <class Tsrv, class Timpl>
inline
fractos::core::future<bool>
impl::Base<Tsrv, Timpl>::destroy_maybe()
{
    if (not _destroyed.test_and_set()) {
        return do_destroy()
            .then([](auto& fut) {
                fut.get();
                return true;
            });
    } else {
        return fractos::core::make_ready_future<bool>(false);
    }
}
