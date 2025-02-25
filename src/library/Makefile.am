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
libfractos_service_compute_cuda_la_SOURCES += srv_service.cpp
libfractos_service_compute_cuda_la_SOURCES += srv_device.cpp
libfractos_service_compute_cuda_la_SOURCES += cuda_device.cpp
libfractos_service_compute_cuda_la_SOURCES += cuda_service.cpp
libfractos_service_compute_cuda_la_LDFLAGS = -version-info 2:0:0


# libfractos_service_compute_cuda_la_CXX = $(NVCC)
# libfractos_service_compute_cuda_la_CXXFLAGS = -std=c++20 -Xcompiler="-fno-rtti -fno-exceptions -fPIC -G -O0 -Werror -fvisibility=hidden" -O0

# libfractos_service_compute_cuda_la_SOURCES += error.cpp

# libfractos_service_compute_cuda_la_SOURCES += lib_service_impl.cpp
# hide all symbols on the client library, only export public API symbols
# libfractos_service_compute_cuda_la_CXXFLAGS = -Xcompiler -fvisibility=hidden


# cuda_program_CXX = $(NVCC)
# cuda_program_CXXFLAGS = -std=c++20 -Xcompiler="-fPIC -Werror" -O0

# CXX = $(NVCC)

# AM_CXXFLAGS =
# AM_CXXFLAGS += -std=c++20 -Xcompiler="-fno-rtti -fno-exceptions -fPIC -G -O0 -Werror -fvisibility=hidden"

# CPP = $(NVCC)
# AM_CPPFLAGS += -std=c++20 -Xcompiler="-fno-rtti -fno-exceptions -fPIC -G -O0 -Werror -fvisibility=hidden"


# PKG_LIBS =  -L/home/mingxuanyang/fractos/experiments/build/system-production-amd64/install/lib -lfractos-core -lfractos-gns -Xlinker -rpath -Xlinker /home/mingxuanyang/fractos/experiments/build/system-production-amd64/install/lib -lboost_thread  -L/home/mingxuanyang/fractos/experiments/build/system-common-production/install/lib -lfractos-common -Xlinker -rpath -Xlinker /home/mingxuanyang/fractos/experiments/build/system-common-production/install/lib  -lglog  -lboost_program_options

