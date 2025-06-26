# -*- makefile-gmake -*-

SRC_mubench-service-compute-cuda = deps/service-compute-cuda/experiments/mubench

$(call define_experiment,mubench-service-compute-cuda,"CUDA service micro-benchmarks")

$(BUILD_TARGET_mubench-service-compute-cuda): build/system-common
$(BUILD_TARGET_mubench-service-compute-cuda): build/service-compute-cuda
$(BUILD_TARGET_mubench-service-compute-cuda): force


GENERATE_mubench-service-compute-cuda =
GENERATE_mubench-service-compute-cuda += $(SRC_mubench-service-compute-cuda)
GENERATE_mubench-service-compute-cuda += $(GENERATE_copy_all)
GENERATE_mubench-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/install/bin/=dist-amd64/bin/
GENERATE_mubench-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/install/lib/=dist-amd64/lib/
GENERATE_mubench-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/mubench/.libs/=dist-amd64/bin/
GENERATE_mubench-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/mubench/empty_kernel.ptx=dist-amd64/bin/


$(RUN_TARGET_mubench-service-compute-cuda): PREPARE_mubench-service-compute-cuda = "run/mubench-service-compute-cuda/env/exp-*.yaml"


PLOT_mubench-service-compute-cuda = --scan --bench-all
