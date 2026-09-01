ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
compose_override_path = $(if $(filter /%,$(1)),$(1),$(ROOT_DIR)/$(1))
COMPOSE_OVERRIDE_ARG := $(if $(strip $(MATE_COMPOSE_OVERRIDE)),-f $(call compose_override_path,$(strip $(MATE_COMPOSE_OVERRIDE))))
COMPOSE := docker compose --project-directory $(ROOT_DIR) -f $(ROOT_DIR)/compose.yaml $(COMPOSE_OVERRIDE_ARG)
SERVICE := dev

export MATE_SOURCE_DIR ?= $(ROOT_DIR)
export MATE_USER_UID ?= $(shell id -u)
export MATE_USER_GID ?= $(shell id -g)
export MATE_DEV_BASE_IMAGE ?= mate-dev-base:local
export MATE_DEV_BASE_CONTEXT ?= docker-image://$(MATE_DEV_BASE_IMAGE)
export MATE_DEV_IMAGE ?= mate-dev:uid-$(MATE_USER_UID)-gid-$(MATE_USER_GID)
export COMPOSE_PROJECT_NAME ?= mate-$(MATE_USER_UID)

.PHONY: base-build base-pull ensure-base user-build build up shell setup test lint config down

base-build:
	$(COMPOSE) build dev-base

base-pull:
	docker pull "$(MATE_DEV_BASE_IMAGE)"

ensure-base:
	@if ! docker image inspect "$(MATE_DEV_BASE_IMAGE)" >/dev/null 2>&1; then \
		$(MAKE) base-build; \
	fi

user-build: ensure-base
	$(COMPOSE) build $(SERVICE)

build: base-build
	$(MAKE) user-build

up:
	@if ! docker image inspect "$(MATE_DEV_IMAGE)" >/dev/null 2>&1; then \
		$(MAKE) user-build; \
	fi
	$(COMPOSE) up --detach --no-build $(SERVICE)

shell: up
	$(COMPOSE) exec $(SERVICE) zsh

setup: up
	$(COMPOSE) exec $(SERVICE) bash -lc \
		'git submodule update --init --recursive && \
		python -m pip install --no-build-isolation --no-deps -e . -v'

test: up
	$(COMPOSE) exec $(SERVICE) pytest tests/

lint: up
	$(COMPOSE) exec $(SERVICE) pre-commit run -a

config:
	$(COMPOSE) config

down:
	$(COMPOSE) down
