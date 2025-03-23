# -*- makefile-gmake -*-

##################################################
# Build target declaration
#
# Same as src/service-compute-cuda.mak

BUILD_CHOICES_app-compute-cuda = $(BUILD_CHOICES_default)

$(call define_experiment,app-compute-cuda,"compute cuda application")


##################################################
# Build target configuration
#
# Same as src/service-compute-cuda.mak. Note that we use $(BUILD_FLAG_exports)
# instead of $(BUILD_ARGS_service-compute-cuda).

$(BUILD_TARGET_app-compute-cuda): SRC=$(CURDIR)/deps/app-compute-cuda
$(BUILD_TARGET_app-compute-cuda): build/system-common
$(BUILD_TARGET_app-compute-cuda): build/service-compute-cuda
$(BUILD_TARGET_app-compute-cuda): force
$(BUILD_TARGET_app-compute-cuda):
	mkdir -p $@
	cp deps/app-compute-cuda/Makefile $@/Makefile
	PKG_CONFIG_PATH="$(PKG_CONFIG_PATH):$(call pkg_config,service-compute-cuda)" \
	./bin/s-bear "$(SRC)" $(MAKE) -C "$@" --jobs $(JOBS) VPATH="$(SRC)/" $(BUILD_FLAG_exports)


##################################################
# Experiment generation
#
# Edit `src/app-compute-cuda/generate.py` to define the parameters for each of the
# experiments you want to execute, and set any files that need to be copied for
# them (such as program binaries, or configuration files).
#
# The role of `generate.py` is to, first, generate a shell script to launch each
# of the experiments (in `run/app-compute-cuda/jobs/*.sh`) and, second, to generate
# an experiment configuration file that describes what exactly each of the
# experiments has to do (in `run/app-compute-cuda/env/exp-*.yaml`).
#
# The `generate.py` script uses sciexp2-expdef
# (https://sciexp2-expdef.readthedocs.io).
#
# Edit `GENERATE_app-compute-cuda` to set the arguments passed to
# `src/app-compute-cuda/generate.py`.

GENERATE_app-compute-cuda =
GENERATE_app-compute-cuda += src/app-compute-cuda
GENERATE_app-compute-cuda += $(GENERATE_copy_all)
GENERATE_app-compute-cuda += --copy src/env-cluster-localhost.inc.yaml=env/
# GENERATE_app-compute-cuda += --copy src/env-cluster-quokka.inc.yaml=env/
GENERATE_app-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/install/bin/=dist-amd64/bin/
GENERATE_app-compute-cuda += --copy $(BUILD_TARGET_service-compute-cuda)/install/lib/=dist-amd64/lib/
GENERATE_app-compute-cuda += --copy $(BUILD_TARGET_app-compute-cuda)/simple=dist-amd64/bin/


##################################################
# Experiment execution
#
# Edit `src/image-*-deploy.def` to add any files and packages you need to execute
# your benchmark (avoid development files/packages to make images smaller).
#
# Each experiment script will use `src/run.py`, which does all the heavy-lifting
# given a specific experiment configuration file (both of which are generated
# above).
#
# The `src/run.py` script uses sciexp2-exprun
# (https://sciexp2-exprun.readthedocs.io).
#
# Relevant make rules:
# - run/app-compute-cuda: execute all experiments defined here

run/app-compute-cuda: PREPARE_app-compute-cuda = "run/app-compute-cuda/env/exp-*.yaml"


##################################################
# Results evaluation
#
# Edit `src/app-compute-cuda/plot.py` to define how results are evaluated and
# plotted.
#
# The script uses sciexp2-expdata (https://sciexp2-expdata.readthedocs.io).
#
# Relevant make rules:
# - plot/app-template: evaluate the benchmarks
#
# Relevant variables:
# - PLOT_app-compute-cuda: default arguments to `src/app-compute-cuda/plot.py`

PLOT_app-compute-cuda = --scan --bench-all
