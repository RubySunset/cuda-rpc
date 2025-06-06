# -*- makefile-gmake -*-

$(call define_experiment_begin,mubench-service-compute-cuda,"CUDA service micro-benchmarks")

$(BUILD_TARGET_this): build/system-common
$(BUILD_TARGET_this): build/service-compute-cuda
$(BUILD_TARGET_this): force


GENERATE_this =
GENERATE_this += $$(SRC)
GENERATE_this += $(GENERATE_copy_all)
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/install/bin/=dist-amd64/bin/
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/install/lib/=dist-amd64/lib/
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/mubench/.libs/=dist-amd64/bin/
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/mubench/empty_kernel.ptx=dist-amd64/bin/

$(GENERATE_TARGET_this): SRC = $(CURDIR)/deps/service-compute-cuda/mubench/experiments


$(RUN_TARGET_this): PREPARE_this = "run/mubench-service-compute-cuda/env/exp-*.yaml"


PLOT_this = --scan --bench-all

$(call define_experiment_end,mubench-service-compute-cuda)
