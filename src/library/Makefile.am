lib_LTLIBRARIES =

AM_CPPFLAGS =
AM_CPPFLAGS += -I$(srcdir)/../include
AM_CPPFLAGS += $(PKG_CPPFLAGS)
AM_CPPFLAGS += $(CUDA_INCLUDE_FLAGS)

AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)
AM_LDFLAGS += $(CUDA_LIBS)

nobase_include_HEADERS = function.inc.hpp
nobase_include_HEADERS += function_impl.hpp

lib_LTLIBRARIES += libfractos-service-compute-cuda.la
libfractos_service_compute_cuda_la_SOURCES =
libfractos_service_compute_cuda_la_SOURCES += error.cpp
libfractos_service_compute_cuda_la_SOURCES += function.cpp
libfractos_service_compute_cuda_la_SOURCES += module.cpp
libfractos_service_compute_cuda_la_SOURCES += memory.cpp
libfractos_service_compute_cuda_la_SOURCES += context.cpp
libfractos_service_compute_cuda_la_SOURCES += device.cpp
libfractos_service_compute_cuda_la_SOURCES += service.cpp
libfractos_service_compute_cuda_la_LDFLAGS = -version-info 2:0:0

