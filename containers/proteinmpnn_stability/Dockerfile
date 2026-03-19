# ProteinMPNN-ddG stability filter
# Extends the vendor image with autoantibody wrapper
# Build context: repo root

FROM ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base

LABEL maintainer="briney@scripps.edu"
LABEL description="autoantibody/proteinmpnn_stability — ProteinMPNN-ddG fold stability filter"

# Install autoantibody package on top of vendor image
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Copy tool scripts
COPY tools/ ./tools/

ENV AUTOANTIBODY_CONTAINER=1
