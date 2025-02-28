lib_LTLIBRARIES =

AM_CPPFLAGS =
AM_CPPFLAGS += -I$(srcdir)/../include
AM_CPPFLAGS += $(PKG_CPPFLAGS)
AM_CPPFLAGS += $(CUDA_INCLUDE_FLAGS)

AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)
AM_LDFLAGS += $(CUDA_LIBS)


lib_LTLIBRARIES += libfractos-service-compute-cuda.la
libfractos_service_compute_cuda_la_SOURCES =
libfractos_service_compute_cuda_la_SOURCES += error.cpp
libfractos_service_compute_cuda_la_SOURCES += srv_service.cpp
libfractos_service_compute_cuda_la_SOURCES += srv_device.cpp
libfractos_service_compute_cuda_la_SOURCES += srv_context.cpp
libfractos_service_compute_cuda_la_SOURCES += srv_memory.cpp
libfractos_service_compute_cuda_la_SOURCES += cuda_memory.cpp
libfractos_service_compute_cuda_la_SOURCES += cuda_context.cpp
libfractos_service_compute_cuda_la_SOURCES += cuda_device.cpp
libfractos_service_compute_cuda_la_SOURCES += cuda_service.cpp
libfractos_service_compute_cuda_la_LDFLAGS = -version-info 2:0:0

