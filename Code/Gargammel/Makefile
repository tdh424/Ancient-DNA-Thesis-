SHELL := /bin/bash

ROOT_DIR := $(CURDIR)
PACKAGE_DIR := $(ROOT_DIR)/Package
OVERRIDES_SCRIPT := $(ROOT_DIR)/scripts/apply_overrides.sh
GARGAMMEL_REPO ?= https://github.com/grenaud/gargammel.git
LIBGAB_REPO ?= https://github.com/grenaud/libgab.git
BAMTOOLS_REPO ?= https://github.com/pezmaster31/bamtools.git

.PHONY: help show-package-path setup-package download-package download-deps apply-overrides build-package verify-package clean-package

help:
	@printf '%s\n' \
		'Gargammel package targets (run from Ancient-DNA/Code/Gargammel):' \
		'  make setup-package      Clone Gargammel into Package/, apply local overrides, and build it' \
		'  make download-package   Clone Gargammel into Package/ if it is missing' \
		'  make download-deps      Clone libgab/ and bamtools/ into Package/ if they are missing' \
		'  make apply-overrides    Reapply the local macOS/path fixes to Package/' \
		'  make build-package      Build the Package/ copy using its Makefile' \
		'  make verify-package     Check that gargammel.pl and core binaries exist' \
		'  make clean-package      Remove the generated Package/ directory' \
		'  make show-package-path  Print the Package/ directory path'

show-package-path:
	@printf '%s\n' "$(PACKAGE_DIR)"

download-package:
	@if [ -d "$(PACKAGE_DIR)" ]; then \
		echo "Package already exists: $(PACKAGE_DIR)"; \
	else \
		git clone --recursive "$(GARGAMMEL_REPO)" "$(PACKAGE_DIR)"; \
	fi

download-deps: download-package
	@if [ -d "$(PACKAGE_DIR)/libgab" ]; then \
		echo "libgab already exists: $(PACKAGE_DIR)/libgab"; \
	else \
		git clone --depth 1 --recursive "$(LIBGAB_REPO)" "$(PACKAGE_DIR)/libgab"; \
	fi
	@if [ -d "$(PACKAGE_DIR)/bamtools" ]; then \
		echo "bamtools already exists: $(PACKAGE_DIR)/bamtools"; \
	else \
		git clone --recursive "$(BAMTOOLS_REPO)" "$(PACKAGE_DIR)/bamtools"; \
	fi

apply-overrides: download-deps
	@"$(OVERRIDES_SCRIPT)" "$(PACKAGE_DIR)"

build-package: apply-overrides
	@$(MAKE) -C "$(PACKAGE_DIR)"

setup-package: build-package
	@$(MAKE) verify-package

verify-package:
	@test -f "$(PACKAGE_DIR)/gargammel.pl"
	@test -x "$(PACKAGE_DIR)/src/fragSim"
	@test -x "$(PACKAGE_DIR)/src/deamSim"
	@test -x "$(PACKAGE_DIR)/src/adptSim"
	@test -x "$(PACKAGE_DIR)/src/fasta2fastas"
	@echo "Verified Gargammel package build under $(PACKAGE_DIR)"

clean-package:
	@rm -rf "$(PACKAGE_DIR)"
	@echo "Removed generated Package/ directory."
