lib_LTLIBRARIES =


AM_CPPFLAGS =
AM_CPPFLAGS += -I$(top_srcdir)/../include
AM_CPPFLAGS += $(PKG_CPPFLAGS)


AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)


lib_LTLIBRARIES += libfractos-service-compute-cuda.la

libfractos_service_compute_cuda_la_SOURCES =
libfractos_service_compute_cuda_la_SOURCES += error.cpp
libfractos_service_compute_cuda_la_SOURCES += service.cpp
# hide all symbols on the client library, only export public API symbols
libfractos_service_compute_cuda_la_CXXFLAGS = -fvisibility=hidden
