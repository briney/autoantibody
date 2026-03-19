# Rosetta flex-ddG oracle
# Extends the Rosetta image with Python 3.12 + autoantibody
# Build context: repo root

FROM rosettacommons/rosetta

LABEL maintainer="briney@scripps.edu"
LABEL description="autoantibody/flex_ddg — Rosetta flex-ddG binding ddG oracle"

# Install Python 3.12 on top of Rosetta image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies for tool scripts
RUN pip install --no-cache-dir biopython numpy

# Copy tool scripts (standalone, no autoantibody install needed)
COPY tools/ ./tools/

ENV AUTOANTIBODY_CONTAINER=1
