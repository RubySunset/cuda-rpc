bin_PROGRAMS =
# AC_LANG(CUDA)

# CXX = $(NVCC)
AM_CXXFLAGS =
AM_CXXFLAGS += -std=c++20 -O0 

# CPP = $(NVCC)



AM_CPPFLAGS =
AM_CPPFLAGS += -I$(srcdir)/../include
AM_CPPFLAGS += -I$(srcdir)/../library
AM_CPPFLAGS += $(PKG_CPPFLAGS)
AM_CPPFLAGS += $(CUDA_INCLUDE_FLAGS)
AM_CPPFLAGS += -std=c++20  -O0 

# PKG_LIBS =  -L/home/mingxuanyang/fractos/experiments/build/system-production-amd64/install/lib -lfractos-core -lfractos-gns -Xlinker -rpath -Xlinker /home/mingxuanyang/fractos/experiments/build/system-production-amd64/install/lib -lboost_thread  -L/home/mingxuanyang/fractos/experiments/build/system-common-production/install/lib -lfractos-common -Xlinker -rpath -Xlinker /home/mingxuanyang/fractos/experiments/build/system-common-production/install/lib  -lglog  -lboost_program_options

AM_LDFLAGS =
AM_LDFLAGS += $(PKG_LDFLAGS)
AM_LDFLAGS += $(PKG_LIBS)
AM_LDFLAGS += $(CUDA_LIBS)


bin_PROGRAMS += fractos-service-compute-cuda

fractos_service_compute_cuda_SOURCES =
fractos_service_compute_cuda_SOURCES += main.cpp
