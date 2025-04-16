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
nodist_libfractos_service_compute_cuda_wrapper_la_SOURCES = default.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES =
libfractos_service_compute_cuda_wrapper_la_SOURCES += lib.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += initialization.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += version.cpp
libfractos_service_compute_cuda_wrapper_la_SOURCES += module.cpp
libfractos_service_compute_cuda_wrapper_la_LDFLAGS =
libfractos_service_compute_cuda_wrapper_la_LDFLAGS += $(AM_LDFLAGS)
libfractos_service_compute_cuda_wrapper_la_LDFLAGS += ../library/libfractos-service-compute-cuda.la

CLEANFILES = default.cpp
BUILT_SOURCES =
BUILT_SOURCES += default.cpp

default.cpp: $(abs_srcdir)/cuda-generate-funcs.sh
default.cpp: $(LIBCUDA)
default.cpp:
	$(abs_srcdir)/cuda-generate-funcs.sh $(LIBCUDA) $@

install-exec-hook:
	$(RM) $(libdir)/libcuda.so.1
	ln -s $(libdir)/libfractos-service-compute-cuda-wrapper.so $(libdir)/libcuda.so.1
