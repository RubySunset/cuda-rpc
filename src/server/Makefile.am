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
fractos_service_compute_cuda_SOURCES += main.cpp
fractos_service_compute_cuda_SOURCES += common.cpp
fractos_service_compute_cuda_SOURCES += service.cpp
fractos_service_compute_cuda_SOURCES += device.cpp
fractos_service_compute_cuda_SOURCES += context.cpp
fractos_service_compute_cuda_SOURCES += stream.cpp
fractos_service_compute_cuda_SOURCES += event.cpp
fractos_service_compute_cuda_SOURCES += module.cpp
fractos_service_compute_cuda_SOURCES += library.cpp
fractos_service_compute_cuda_SOURCES += kernel.cpp
fractos_service_compute_cuda_SOURCES += memory.cpp
fractos_service_compute_cuda_SOURCES += function.cpp
fractos_service_compute_cuda_SOURCES += cublas.cpp
fractos_service_compute_cuda_SOURCES += cuda_host_cb_manager.cpp
fractos_service_compute_cuda_SOURCES += ../library/error.cpp
fractos_service_compute_cuda_SOURCES += ../library/wire.cpp

nodist_fractos_service_compute_cuda_SOURCES =
nodist_fractos_service_compute_cuda_SOURCES += cublas-codegen.cpp

.cublas-codegen-stamp: $(abs_srcdir)/../../bin/cublas-codegen.py
	uv run --with libclang $(abs_srcdir)/../../bin/cublas-codegen.py \
		--cublas-header /usr/local/cuda/include/cublas_v2.h \
		--cublas-manual-file $(abs_srcdir)/../wrapper/cublas-manual.cpp \
		--wrapper-dir ../wrapper \
		--server-dir .
	touch $@

CLEANFILES =
CLEANFILES += cublas-codgen.cpp .cublas-codegen-stamp
BUILT_SOURCES =
BUILT_SOURCES += cublas-codegen.cpp

cublas-codegen.cpp: .cublas-codegen-stamp
