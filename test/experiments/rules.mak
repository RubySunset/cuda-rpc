# -*- makefile-gmake -*-

$(call define_experiment_begin,test-service-compute-cuda,"CUDA service regression tests")

$(BUILD_TARGET_this): build/system-common
$(BUILD_TARGET_this): build/service-compute-cuda
$(BUILD_TARGET_this): force


GENERATE_this =
GENERATE_this += $$(SRC)
GENERATE_this += $(GENERATE_copy_all)
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/install/bin/=dist-amd64/bin/
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/install/lib/=dist-amd64/lib/
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/test/.libs/=dist-amd64/bin/
GENERATE_this += --copy $(BUILD_TARGET_service-compute-cuda)/test/module.ptx=dist-amd64/bin/

ifneq ($(MAKECMDGOALS),sync/$(TARGET_this))
$(GENERATE_TARGET_this): build-force/service-compute-cuda
endif
$(GENERATE_TARGET_this): SRC = $(CURDIR)/deps/service-compute-cuda/test/experiments


$(RUN_TARGET_this): PREPARE_this = "run/test-service-compute-cuda/env/exp-*.yaml"

$(call define_experiment_end,test-service-compute-cuda)
