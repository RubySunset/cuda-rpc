bin_PROGRAMS =

AM_CPPFLAGS =
AM_CPPFLAGS += -I$(srcdir)/../include -I$(srcdir)/../library
AM_CPPFLAGS += $(PKG_CPPFLAGS)
AM_CPPFLAGS += $(CUDA_INCLUDE_FLAGS)

AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)
AM_LDFLAGS += $(CUDA_LIBS)


bin_PROGRAMS += fractos-service-compute-cuda

fractos_service_compute_cuda_SOURCES =
fractos_service_compute_cuda_SOURCES += srv_service.cpp
fractos_service_compute_cuda_SOURCES += srv_device.cpp
fractos_service_compute_cuda_SOURCES += srv_context.cpp
fractos_service_compute_cuda_SOURCES += srv_stream.cpp
fractos_service_compute_cuda_SOURCES += srv_event.cpp
fractos_service_compute_cuda_SOURCES += srv_module.cpp
fractos_service_compute_cuda_SOURCES += srv_memory.cpp
fractos_service_compute_cuda_SOURCES += function.cpp
fractos_service_compute_cuda_SOURCES += main.cpp
fractos_service_compute_cuda_SOURCES += ../library/error.cpp
fractos_service_compute_cuda_SOURCES += ../library/wire.cpp
