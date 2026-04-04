SHELL := /usr/bin/env bash

IMAGE_NAME ?= llm-training
DOCKER_TAG ?= latest
NAMESPACE ?= llm-training
K8S_DIR ?= k8s

include Makefile.docker
include Makefile.k8s

.PHONY: help setup namespace build push build-push apply-pvc deploy-training deploy-eval deploy-all run-training run-eval run-all logs-training logs-eval status describe-training describe-eval clean clean-all

help: ## Show available commands
	@echo "LLM Training Kubernetes CLI"
	@echo
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Configurable variables:"
	@echo "  IMAGE_NAME=$(IMAGE_NAME)"
	@echo "  DOCKER_TAG=$(DOCKER_TAG)"
	@echo "  NAMESPACE=$(NAMESPACE)"
	@echo "  K8S_DIR=$(K8S_DIR)"
