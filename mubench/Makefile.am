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

noinst_DATA = empty_kernel.ptx

noinst_PROGRAMS += mubench-memory_alloc
mubench_memory_alloc_SOURCES = memory_alloc.cpp

noinst_PROGRAMS += mubench-kernel_launch
mubench_kernel_launch_SOURCES = kernel_launch.cu
mubench_kernel_launch_LINK = $(CXXLINK)

NVCCCOMPILE = $(NVCC) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS) \
	-diag-suppress 970 -diag-suppress 550 -diag-suppress 186

.cu.ptx:
	$(AM_V_CXX)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.o$$||'`;\
	$(filter-out -Wall -Werror -rdynamic -fvisibility=hidden,$(NVCCCOMPILE)) -ptx -MT $@ -MMD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po

.cu.o:
	$(AM_V_CXX)depbase=`echo $@ | sed 's|[^/]*$$|$(DEPDIR)/&|;s|\.o$$||'`;\
	$(filter-out -Wall -Werror -rdynamic -fvisibility=hidden,$(NVCCCOMPILE)) -MT $@ -MMD -MP -MF $$depbase.Tpo -c -o $@ $< &&\
	$(am__mv) $$depbase.Tpo $$depbase.Po
