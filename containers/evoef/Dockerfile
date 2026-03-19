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

# Install autoantibody package
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# Copy tool scripts
COPY tools/ ./tools/

# Copy compiled EvoEF binary + data files
COPY --from=builder /opt/evoef/EvoEF /usr/local/bin/EvoEF
COPY --from=builder /opt/evoef/data /opt/evoef/data

ENV AUTOANTIBODY_CONTAINER=1
ENV EVOEF_BINARY=/usr/local/bin/EvoEF
# EvoEF expects its data dir relative to binary or via EVOEF_DATA_DIR
ENV EVOEF_DATA_DIR=/opt/evoef/data
