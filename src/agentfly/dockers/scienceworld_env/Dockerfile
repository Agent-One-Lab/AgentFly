FROM python:3.11-slim
# Install Java
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone and install ScienceWorld
RUN cd /srv && \
    git clone https://github.com/allenai/ScienceWorld.git && \
    cd ScienceWorld && \
    pip install .

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scienceworld_server.py /srv/
COPY start.sh /srv/
RUN chmod +x /srv/start.sh

WORKDIR /srv

# Expose the default port
EXPOSE 2700

ENTRYPOINT ["/srv/start.sh"]
