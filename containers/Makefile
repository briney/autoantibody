# Makefile for building autoantibody Docker images
# Run from repo root: make -C containers build-all
# Or: make -C containers build-evoef

REPO_ROOT := $(shell dirname $(realpath $(MAKEFILE_LIST)))/../
IMAGE_PREFIX := autoantibody

# Priority tools (this PR)
PRIORITY_TOOLS := evoef stabddg ablms proteinmpnn_stability flex_ddg

# All tools (including deferred)
ALL_TOOLS := $(PRIORITY_TOOLS)

.PHONY: build-all build-base-cpu build-base-gpu test-all clean $(addprefix build-,$(ALL_TOOLS)) $(addprefix test-,$(ALL_TOOLS))

# --- Base images ---

build-base-cpu:
	docker build \
		-f $(REPO_ROOT)/containers/base/Dockerfile.cpu \
		-t $(IMAGE_PREFIX)/base:cpu \
		$(REPO_ROOT)

build-base-gpu:
	docker build \
		-f $(REPO_ROOT)/containers/base/Dockerfile.gpu \
		-t $(IMAGE_PREFIX)/base:gpu \
		$(REPO_ROOT)

# --- Per-tool images ---

build-evoef:
	docker build \
		-f $(REPO_ROOT)/containers/evoef/Dockerfile \
		-t $(IMAGE_PREFIX)/evoef:latest \
		$(REPO_ROOT)

build-stabddg:
	docker build \
		-f $(REPO_ROOT)/containers/stabddg/Dockerfile \
		-t $(IMAGE_PREFIX)/stabddg:latest \
		$(REPO_ROOT)

build-ablms:
	docker build \
		-f $(REPO_ROOT)/containers/ablms/Dockerfile \
		-t $(IMAGE_PREFIX)/ablms:latest \
		$(REPO_ROOT)

build-proteinmpnn_stability:
	docker build \
		-f $(REPO_ROOT)/containers/proteinmpnn_stability/Dockerfile \
		-t $(IMAGE_PREFIX)/proteinmpnn_stability:latest \
		$(REPO_ROOT)

build-flex_ddg:
	docker build \
		-f $(REPO_ROOT)/containers/flex_ddg/Dockerfile \
		-t $(IMAGE_PREFIX)/flex_ddg:latest \
		$(REPO_ROOT)

# --- Aggregate targets ---

build-all: $(addprefix build-,$(PRIORITY_TOOLS))

# --- Smoke tests ---

test-evoef: build-evoef
	docker run --rm $(IMAGE_PREFIX)/evoef:latest python -c "from autoantibody.models import ToolResult; print('ok')"

test-stabddg: build-stabddg
	docker run --rm $(IMAGE_PREFIX)/stabddg:latest python -c "from autoantibody.models import ToolResult; print('ok')"

test-ablms: build-ablms
	docker run --rm $(IMAGE_PREFIX)/ablms:latest python -c "from autoantibody.models import ToolResult; print('ok')"

test-proteinmpnn_stability: build-proteinmpnn_stability
	docker run --rm $(IMAGE_PREFIX)/proteinmpnn_stability:latest python -c "from autoantibody.models import ToolResult; print('ok')"

test-flex_ddg: build-flex_ddg
	docker run --rm $(IMAGE_PREFIX)/flex_ddg:latest python -c "from autoantibody.models import ToolResult; print('ok')"

test-all: $(addprefix test-,$(PRIORITY_TOOLS))

# --- Cleanup ---

clean:
	@for tool in $(ALL_TOOLS); do \
		docker rmi $(IMAGE_PREFIX)/$$tool:latest 2>/dev/null || true; \
	done
	docker rmi $(IMAGE_PREFIX)/base:cpu 2>/dev/null || true
	docker rmi $(IMAGE_PREFIX)/base:gpu 2>/dev/null || true
