#!/usr/bin/env bash
# install.sh — Install all prerequisites for the autoantibody optimization system.
#
# Usage:
#   ./install.sh              # Install everything (required + recommended)
#   ./install.sh --minimal    # Required tools only (minimal Docker containers)
#   ./install.sh --skip-docker  # Skip Docker image pulls
#
# Environment variables:
#   CUDA_VERSION  — CUDA version for PyTorch (default: auto-detected)
#
# All scoring tools (evoef, ablms, graphinity, stabddg, proteinmpnn, flex_ddg)
# run inside Docker containers — they are NOT installed locally.

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

MINIMAL=false
SKIP_DOCKER=false

for arg in "$@"; do
    case "$arg" in
        --minimal) MINIMAL=true ;;
        --skip-docker) SKIP_DOCKER=true ;;
        --help|-h)
            head -10 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
info()  { echo -e "\033[1;34m==>\033[0m \033[1m$*\033[0m"; }
ok()    { echo -e "\033[1;32m  ✓\033[0m $*"; }
warn()  { echo -e "\033[1;33m  !\033[0m $*"; }
fail()  { echo -e "\033[1;31m  ✗\033[0m $*"; exit 1; }

command_exists() { command -v "$1" &>/dev/null; }

# ------------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------------
info "Preflight checks"

command_exists python3 || fail "python3 not found"
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if (( PYTHON_MAJOR < 3 || PYTHON_MINOR < 12 )); then
    fail "Python >= 3.12 required, found $PYTHON_VERSION"
fi
ok "Python $PYTHON_VERSION"

command_exists git || fail "git not found"
ok "git"

if ! $SKIP_DOCKER; then
    command_exists docker || fail "docker not found"
    docker info &>/dev/null || fail "Docker daemon not running or user not in docker group"
    ok "Docker"
fi

# Detect CUDA version for PyTorch
CUDA_VERSION="${CUDA_VERSION:-}"
if [ -z "$CUDA_VERSION" ]; then
    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//' | tr -d '.')
        # Normalize: 128 -> cu128, 124 -> cu124, etc.
        if [ ${#CUDA_VERSION} -eq 2 ]; then
            CUDA_VERSION="${CUDA_VERSION}0"
        fi
        CUDA_VERSION="cu${CUDA_VERSION}"
        ok "CUDA detected: $CUDA_VERSION"
    elif command_exists nvidia-smi; then
        # Fall back to driver-reported CUDA version
        CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | tr -d '.')
        if [ ${#CUDA_VER} -eq 2 ]; then
            CUDA_VER="${CUDA_VER}0"
        fi
        CUDA_VERSION="cu${CUDA_VER}"
        ok "CUDA detected from driver: $CUDA_VERSION"
    else
        CUDA_VERSION="cpu"
        warn "No CUDA detected — PyTorch will use CPU only"
    fi
fi

echo ""

# ------------------------------------------------------------------
# 1. Virtual environment
# ------------------------------------------------------------------
info "Setting up virtual environment"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
else
    ok "Venv already exists at $VENV_DIR"
fi

# Activate venv for this script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
ok "pip upgraded"
echo ""

# ------------------------------------------------------------------
# 2. Project dependencies
# ------------------------------------------------------------------
info "Installing project dependencies"

pip install -e ".[dev]" --quiet
ok "autoantibody + dev deps installed"
echo ""

# ------------------------------------------------------------------
# 3. PyTorch
# ------------------------------------------------------------------
info "Installing PyTorch ($CUDA_VERSION)"

if python -c "import torch" &>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    ok "PyTorch already installed: $TORCH_VER"
else
    if [ "$CUDA_VERSION" = "cpu" ]; then
        pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
    else
        pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_VERSION}" --quiet
    fi
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    ok "PyTorch installed: $TORCH_VER"
fi

# Verify CUDA availability
if [ "$CUDA_VERSION" != "cpu" ]; then
    if python -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
        GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
        ok "CUDA available: $GPU_NAME"
    else
        warn "PyTorch installed but CUDA not available — ML scorers will be slow"
    fi
fi
echo ""

# ------------------------------------------------------------------
# 4. Docker images (containerized scoring tools)
# ------------------------------------------------------------------
if ! $SKIP_DOCKER; then
    info "Building containerized scoring tools"

    if $MINIMAL; then
        # Minimal: build only fast-tier tools
        DOCKER_TARGETS="build-evoef"
    else
        # Full: build all priority tools
        DOCKER_TARGETS="build-all"
    fi

    if [ -f "$SCRIPT_DIR/containers/Makefile" ]; then
        make -C "$SCRIPT_DIR/containers" $DOCKER_TARGETS
        ok "Docker images built ($DOCKER_TARGETS)"
    else
        warn "containers/Makefile not found — skipping Docker builds"
    fi
    echo ""
fi

# ------------------------------------------------------------------
# 5. Test data
# ------------------------------------------------------------------
info "Downloading test data"

TEST_DATA_DIR="$SCRIPT_DIR/tests/data"
mkdir -p "$TEST_DATA_DIR"

if [ -f "$TEST_DATA_DIR/1N8Z.pdb" ]; then
    ok "PDB 1N8Z already downloaded"
else
    curl -fsSL -o "$TEST_DATA_DIR/1N8Z.pdb" "https://files.rcsb.org/download/1N8Z.pdb"
    ok "PDB 1N8Z downloaded ($(wc -l < "$TEST_DATA_DIR/1N8Z.pdb") lines)"
fi
echo ""

# ------------------------------------------------------------------
# 6. Verification
# ------------------------------------------------------------------
info "Verifying installation"

PASS=0
TOTAL=0

check() {
    TOTAL=$((TOTAL + 1))
    if eval "$2" &>/dev/null; then
        ok "$1"
        PASS=$((PASS + 1))
    else
        warn "FAILED: $1"
    fi
}

check "autoantibody package"     "python -c 'import autoantibody'"
check "pydantic"                 "python -c 'import pydantic'"
check "biopython"                "python -c 'import Bio'"
check "numpy"                    "python -c 'import numpy'"
check "typer"                    "python -c 'import typer'"
check "PyTorch"                  "python -c 'import torch'"
check "PDB 1N8Z"                 "test -f '$TEST_DATA_DIR/1N8Z.pdb'"

if ! $SKIP_DOCKER; then
    check "evoef container"          "docker image inspect autoantibody/evoef:latest"
    if ! $MINIMAL; then
        check "stabddg container"    "docker image inspect autoantibody/stabddg:latest"
        check "ablms container"      "docker image inspect autoantibody/ablms:latest"
        check "proteinmpnn container" "docker image inspect autoantibody/proteinmpnn_stability:latest"
        check "flex_ddg container"   "docker image inspect autoantibody/flex_ddg:latest"
    fi
fi

echo ""
info "Verification: $PASS/$TOTAL checks passed"

if [ "$PASS" -lt "$TOTAL" ]; then
    warn "Some checks failed — review warnings above"
else
    ok "All checks passed!"
fi

echo ""
info "Setup complete!"
echo ""
echo "  To activate the environment:"
echo "    source $VENV_DIR/bin/activate"
echo ""
