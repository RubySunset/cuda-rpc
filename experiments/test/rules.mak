# -*- makefile-gmake -*-

SRC_test-service-compute-cuda = deps/service-compute-cuda/experiments/test

$(call define_experiment,test-service-compute-cuda,"CUDA service regression tests")

$(BUILD_TARGET_test-service-compute-cuda): build/system-common
$(BUILD_TARGET_test-service-compute-cuda): build/service-compute-cuda
$(BUILD_TARGET_test-service-compute-cuda): force


GENERATE_test-service-compute-cuda =
GENERATE_test-service-compute-cuda += $(SRC_test-service-compute-cuda)
GENERATE_test-service-compute-cuda += $(GENERATE_copy_all)
GENERATE_test-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/install/bin/=dist-amd64/bin/
GENERATE_test-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/install/lib/=dist-amd64/lib/
GENERATE_test-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/test/.libs/=dist-amd64/bin/
GENERATE_test-service-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/test/module.ptx=dist-amd64/bin/


run/test-service-compute-cuda: PREPARE_test-service-compute-cuda = "run/test-service-compute-cuda/env/exp-*.yaml"
run/test-service-compute-cuda: build-force/service-compute-cuda
