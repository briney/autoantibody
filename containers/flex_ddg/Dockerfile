# Rosetta flex-ddG oracle
# Extends the Rosetta image with tool script dependencies
# Build context: repo root

FROM rosettacommons/rosetta

LABEL maintainer="briney@scripps.edu"
LABEL description="autoantibody/flex_ddg — Rosetta flex-ddG binding ddG oracle"

# Install pip for the system Python 3.8
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies (pinned to last versions supporting 3.8)
RUN pip install --no-cache-dir "biopython<1.83" "numpy<1.25"

# Copy tool scripts (standalone, no autoantibody install needed)
COPY tools/ ./tools/

ENV AUTOANTIBODY_CONTAINER=1
