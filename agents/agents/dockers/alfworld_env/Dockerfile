# ALFWorld HTTP Environment Docker Image
# Multi-stage build for smaller final image
FROM continuumio/miniconda3:latest as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create conda environment
COPY requirements.txt /tmp/requirements.txt
RUN conda create -n alfworld python=3.10 -y && \
    echo "source activate alfworld" > ~/.bashrc
ENV PATH /opt/conda/envs/alfworld/bin:$PATH

# Install Python dependencies
RUN /opt/conda/envs/alfworld/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Install ALFWorld and dependencies with proper library support
RUN /opt/conda/envs/alfworld/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    /opt/conda/envs/alfworld/bin/pip install --no-cache-dir alfworld[full]

# Install updated GCC libraries for compatibility
RUN conda install -n alfworld -c conda-forge libstdcxx-ng=13.2.0 gcc_linux-64 gxx_linux-64 -y

# Download ALFWorld data during build
RUN mkdir -p /root/.cache/alfworld && \
    /opt/conda/envs/alfworld/bin/python -c "import alfworld; alfworld.agents.environment.alfred_tw_env.download_extra_content()" || true

# Alternative: Download data using alfworld-download command
RUN /opt/conda/envs/alfworld/bin/alfworld-download

# Verify data was downloaded
RUN ls -la /root/.cache/alfworld/ && \
    echo "ALFWorld data downloaded successfully during build"

# Production stage
FROM continuumio/miniconda3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV ALFWORLD_CONFIG=/srv/base_config.yaml

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy the conda environment from builder
COPY --from=builder /opt/conda/envs/alfworld /opt/conda/envs/alfworld

# Copy ALFWorld data from builder
COPY --from=builder /root/.cache/alfworld /root/.cache/alfworld

# Set PATH
ENV PATH /opt/conda/envs/alfworld/bin:$PATH

# Set working directory
WORKDIR /srv

# Copy application files
COPY alfworld_http_server.py /srv/
COPY base_config.yaml /srv/
COPY start.sh /srv/

# Make start script executable
RUN chmod +x /srv/start.sh

# Ensure data directory exists
RUN mkdir -p /root/.cache/alfworld

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint and default command
ENTRYPOINT ["/srv/start.sh"]
CMD ["base_config.yaml"] 