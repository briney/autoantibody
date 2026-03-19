# EvoEF scorer: multi-stage build compiles from source
# Build context: repo root

# --- Build stage: compile EvoEF ---
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ git && \
    rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/tommyhuangthu/EvoEF.git /opt/evoef && \
    cd /opt/evoef && \
    g++ -O3 -ffast-math -o EvoEF src/*.cpp

# --- Runtime stage ---
FROM python:3.12-slim

LABEL maintainer="briney@scripps.edu"
LABEL description="autoantibody/evoef — EvoEF physics-based binding ddG"

WORKDIR /app

# Install biopython for PDB structure validation
RUN pip install --no-cache-dir biopython

# Copy tool scripts (standalone, no autoantibody install needed)
COPY tools/ ./tools/

# Copy compiled EvoEF binary + library files (must be colocated;
# EvoEF resolves library/ relative to the binary's directory via argv[0])
COPY --from=builder /opt/evoef/EvoEF /opt/evoef/EvoEF
COPY --from=builder /opt/evoef/library /opt/evoef/library

ENV AUTOANTIBODY_CONTAINER=1
ENV EVOEF_BINARY=/opt/evoef/EvoEF
ENV PATH="/opt/evoef:${PATH}"
