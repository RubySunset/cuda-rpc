noinst_PROGRAMS =

AM_CPPFLAGS =
AM_CPPFLAGS += $(PKG_CPPFLAGS)
AM_CPPFLAGS += $(CUDA_INCLUDE_FLAGS)
AM_CPPFLAGS += -I$(top_srcdir)/src/include

AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)
AM_LDFLAGS += $(top_builddir)/src/library/libfractos-service-compute-cuda.la
AM_LDFLAGS += $(CUDA_LIBS)

noinst_DATA = module.ptx

module.ptx: module.cu
	$(NVCC) -ptx $(CPPFLAGS) $$(subst -Werror,-Werror all-warnings,$(CXXFLAGS)) -o $@ $< $(LDFLAGS)

noinst_PROGRAMS += test-service-init
test_service_init_SOURCES = test-service-init.cpp

noinst_PROGRAMS += test-service-memory
test_service_memory_SOURCES = test-service-memory.cpp

noinst_PROGRAMS += test-stream
test_stream_SOURCES = test-stream.cpp
