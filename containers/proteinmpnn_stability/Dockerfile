# ProteinMPNN-ddG stability filter
# Extends the vendor image with autoantibody wrapper
# Build context: repo root

FROM ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base

LABEL maintainer="briney@scripps.edu"
LABEL description="autoantibody/proteinmpnn_stability — ProteinMPNN-ddG fold stability filter"

# Activate micromamba conda environment for RUN commands
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install biopython for PDB structure validation
WORKDIR /app
RUN pip install --no-cache-dir biopython

# Copy tool scripts
COPY tools/ ./tools/

ENV AUTOANTIBODY_CONTAINER=1
ENV JAX_PLATFORMS=cpu
