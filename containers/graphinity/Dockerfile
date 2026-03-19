# Graphinity equivariant GNN scorer: CPU image with old PyTorch stack
# Requires micromamba for OpenBabel (atom typing dependency).
# Build context: repo root

FROM mambaorg/micromamba:1.5-jammy

LABEL maintainer="briney@scripps.edu"
LABEL description="autoantibody/graphinity — Graphinity equivariant GNN binding ddG"

# --- Root: system deps + EvoEF compilation + repo clones ---
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ git && \
    rm -rf /var/lib/apt/lists/*

# Compile EvoEF (used for mutant PDB generation inside the container)
RUN git clone --depth 1 https://github.com/tommyhuangthu/EvoEF.git /opt/evoef && \
    cd /opt/evoef && \
    g++ -O3 -ffast-math -o EvoEF src/*.cpp

# Clone Graphinity repo (includes pre-trained checkpoints ~35MB)
RUN git clone https://github.com/oxpig/Graphinity.git /opt/graphinity

# --- Micromamba user: conda + pip packages ---
USER $MAMBA_USER
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Python 3.8 + OpenBabel + scientific deps via conda-forge
RUN micromamba install -n base -y \
    python=3.8 \
    openbabel=3.1.1 \
    biopython \
    numpy=1.21 \
    pandas=1.3 \
    scipy \
    pyarrow \
    tqdm \
    pyyaml \
    -c conda-forge && \
    micromamba clean --all --yes

# BioPandas via pip (not always on conda-forge)
RUN pip install --no-cache-dir biopandas

# PyTorch 1.8 CPU (Graphinity requires this specific version)
RUN pip install --no-cache-dir \
    torch==1.8.0+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# PyTorch Geometric 1.6.3 + extensions (matching torch 1.8 CPU)
RUN pip install --no-cache-dir \
    torch-scatter==2.0.8 \
    torch-sparse==0.6.12 \
    torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 \
    -f https://data.pyg.org/whl/torch-1.8.0+cpu.html && \
    pip install --no-cache-dir \
    torch-geometric==1.6.3 \
    pytorch-lightning==1.2.10

# --- Root: final setup ---
USER root
WORKDIR /app

# Copy tool scripts (standalone, no autoantibody install needed)
COPY tools/ ./tools/

# Ensure graphinity source is importable (for direct Python usage)
RUN chmod -R a+r /opt/graphinity

ENV AUTOANTIBODY_CONTAINER=1
ENV EVOEF_BINARY=/opt/evoef/EvoEF
ENV GRAPHINITY_DIR=/opt/graphinity
ENV PATH="/opt/evoef:${PATH}"
