# ablms scorer: GPU image with PyTorch + ablms
# Build context: repo root

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL maintainer="briney@scripps.edu"
LABEL description="autoantibody/ablms — Antibody language model scoring"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        gcc g++ git && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CUDA 12.4)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install ablms
RUN pip install --no-cache-dir git+https://github.com/briney/ablms.git

# Copy tool scripts (standalone, no autoantibody install needed)
COPY tools/ ./tools/

# Clean up build deps
RUN apt-get purge -y gcc g++ python3.12-dev && \
    apt-get autoremove -y

ENV AUTOANTIBODY_CONTAINER=1
