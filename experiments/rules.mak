# -*- makefile-gmake -*-

$(call define_buildable,service-compute-cuda,"CUDA compute service")

BUILD_service-compute-cuda_production =
BUILD_service-compute-cuda_debug = --enable-debug
BUILD_service-compute-cuda_gprof = --enable-gprof
BUILD_service-compute-cuda_sanitizer = --enable-sanitizer
BUILD_service-compute-cuda_valgrind = --enable-sanitizer


$(call RULE_BEGIN,deps/service-compute-cuda/configure)
deps/service-compute-cuda/configure:
	$(rule-reset)
	mkdir -p $$(dirname $@)
	cd $$(dirname $@) && $(CURDIR)/bin/s-exec ./autogen.sh
	$(rule-commit)


$(call RULE_BEGIN,$(BUILD_TARGET_service-compute-cuda)/Makefile)
$(BUILD_TARGET_service-compute-cuda)/Makefile: build/system-common
$(BUILD_TARGET_service-compute-cuda)/Makefile: deps/service-compute-cuda/configure
$(BUILD_TARGET_service-compute-cuda)/Makefile: SRC=$(CURDIR)/deps/service-compute-cuda
$(BUILD_TARGET_service-compute-cuda)/Makefile:
	$(rule-reset)
	mkdir -p $(BUILD_TARGET_service-compute-cuda)/install
	cd $(BUILD_TARGET_service-compute-cuda) && \
	PKG_CONFIG_PATH="$(PKG_CONFIG_PATH)" \
	$(CURDIR)/bin/s-exec $(SRC)/configure \
		$(BUILD_ARGS_service-compute-cuda) \
		--prefix=$(CURDIR)/$(BUILD_TARGET_service-compute-cuda)/install \
		INSTALL="`which install` -C"
	$(rule-commit)

$(call RULE_BEGIN,$(BUILD_TARGET_service-compute-cuda))
$(BUILD_TARGET_service-compute-cuda): $(BUILD_TARGET_service-compute-cuda)/Makefile
$(BUILD_TARGET_service-compute-cuda): SRC=$(CURDIR)/deps/service-compute-cuda
$(BUILD_TARGET_service-compute-cuda):
	./bin/s-bear "$(SRC)" $(MAKE) -C "$@" --jobs $(JOBS) install
	$(rule-commit)
