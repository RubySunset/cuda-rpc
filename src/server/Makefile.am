bin_PROGRAMS =

AM_CPPFLAGS =
AM_CPPFLAGS += -I$(srcdir)/../include
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
fractos_service_compute_cuda_SOURCES += srv_module.cpp
fractos_service_compute_cuda_SOURCES += srv_memory.cpp
fractos_service_compute_cuda_SOURCES += main.cpp
fractos_service_compute_cuda_LDADD = ../library/libfractos-service-compute-cuda.la
