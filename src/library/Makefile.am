lib_LTLIBRARIES =

# CXX = $(NVCC)
AM_CXXFLAGS =
AM_CXXFLAGS += -std=c++20 -Xcompiler -G -O0 --compiler-options -Werror 

# CPP = $(NVCC)

AM_CPPFLAGS =
AM_CPPFLAGS += -I$(top_srcdir)/../include
AM_CPPFLAGS += $(PKG_CPPFLAGS)
AM_CPPFLAGS += $(CUDA_INCLUDE_FLAGS)
AM_CPPFLAGS += -std=c++20 -Xcompiler -G -O0  --compiler-options -Werror 


# PKG_LIBS =  -L/home/mingxuanyang/fractos/experiments/build/system-production-amd64/install/lib -lfractos-core -lfractos-gns -Xlinker -rpath -Xlinker /home/mingxuanyang/fractos/experiments/build/system-production-amd64/install/lib -lboost_thread  -L/home/mingxuanyang/fractos/experiments/build/system-common-production/install/lib -lfractos-common -Xlinker -rpath -Xlinker /home/mingxuanyang/fractos/experiments/build/system-common-production/install/lib  -lglog  -lboost_program_options


AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)
AM_LDFLAGS += $(CUDA_LIBS)


lib_LTLIBRARIES += libfractos-service-compute-cuda.la

libfractos_service_compute_cuda_la_SOURCES =
libfractos_service_compute_cuda_la_SOURCES += error.cpp
libfractos_service_compute_cuda_la_SOURCES += service.cpp
# hide all symbols on the client library, only export public API symbols
libfractos_service_compute_cuda_la_CXXFLAGS = -fvisibility=hidden
