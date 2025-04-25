lib_LTLIBRARIES =

AM_CPPFLAGS =
AM_CPPFLAGS += -I$(srcdir)/../include
AM_CPPFLAGS += $(PKG_CPPFLAGS)
AM_CPPFLAGS += $(CUDA_INCLUDE_FLAGS)

AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)
AM_LDFLAGS += $(CUDA_LIBS)

lib_LTLIBRARIES += libfractos-service-compute-cuda-wrapper.la
nodist_libfractos_service_compute_cuda_wrapper_la_SOURCES =
libfractos_service_compute_cuda_wrapper_la_SOURCES =
libfractos_service_compute_cuda_wrapper_la_SOURCES += common.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += ../library/wire.cpp

nodist_libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-default.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-lib.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-state.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-error.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-initialization.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-version.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-device.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-primary-context.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-context.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += driver-module.cpp
libfractos_service_compute_cuda_wrapper_la_LDFLAGS =
libfractos_service_compute_cuda_wrapper_la_LDFLAGS += $(AM_LDFLAGS)
libfractos_service_compute_cuda_wrapper_la_LDFLAGS += ../library/libfractos-service-compute-cuda.la

CLEANFILES =
BUILT_SOURCES =

CLEANFILES += driver-default.cpp
BUILT_SOURCES += driver-default.cpp

driver-default.cpp: $(abs_srcdir)/cuda-generate-funcs.sh
driver-default.cpp: $(LIBCUDA)
driver-default.cpp:
	$(abs_srcdir)/cuda-generate-funcs.sh driver $(LIBCUDA) $@

install-exec-hook:
	$(RM) $(libdir)/libcuda.so.1
	ln -s $(libdir)/libfractos-service-compute-cuda-wrapper.so $(libdir)/libcuda.so.1
