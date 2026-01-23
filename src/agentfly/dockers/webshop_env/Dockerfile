FROM ubuntu:latest

# Set working directory
WORKDIR /app

# Install system dependencies including Java
RUN apt-get update && apt-get install -y \
    openjdk-21-jdk \
    wget \
    pkg-config \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Minoconda
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
RUN rm ~/miniconda3/miniconda.sh

# Add Miniconda to PATH
ENV PATH="/root/miniconda3/bin:$PATH"
RUN ~/miniconda3/bin/conda install -y python=3.10
SHELL ["/bin/bash", "-c"]
RUN ~/miniconda3/bin/conda init --all

# # Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model, TODO: may need to change to en_core_web_lg
RUN python -m spacy download en_core_web_sm

# Download dataset
RUN mkdir -p /app/data && \
    cd /app/data && \
    # Download items_human_ins
    gdown https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O && \
    mkdir -p /app/data/full && \
    cd /app/data/full && \
    # gdown https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib -O items_shuffle_1000.json && \
    # gdown https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu -O items_ins_v2_1000.json && \
    # TODO: change to full data after all testing
    gdown https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB && \
    gdown https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi && \
    ls -l

RUN cd /app
# Copy application code and necessary directories
COPY indexes/ /app/indexes/
COPY static/ /app/static/
COPY templates/ /app/templates/
COPY engine.py .
COPY goal.py .
COPY normalize.py .
COPY utils.py .
COPY webshop_simulator_server.py .

# Create and set permissions for temporary directories
RUN mkdir -p /tmp /var/tmp /usr/tmp && \
    chmod 1777 /tmp /var/tmp /usr/tmp && \
    chown -R nobody:nogroup /tmp /var/tmp /usr/tmp && \
    chmod -R 777 /app

# Expose the port the app runs on
EXPOSE 3000

COPY start.sh /app
RUN chmod +x /app/start.sh
# Command to run the application
ENTRYPOINT ["/app/start.sh"]
