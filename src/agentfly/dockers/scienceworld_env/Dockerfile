# bookworm: trixie (the newer default for python:*-slim) has no openjdk-17.
FROM python:3.11.14-slim-bookworm
# Install Java
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ScienceWorld (pinned release) and the server's dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scienceworld_server.py /srv/
COPY start.sh /srv/
RUN chmod +x /srv/start.sh

WORKDIR /srv

# Expose the default port
EXPOSE 2700

ENTRYPOINT ["/srv/start.sh"]
