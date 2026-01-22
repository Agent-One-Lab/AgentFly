FROM python:3.10-slim

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY python_http_server.py /srv/
COPY snippet_runner.py /srv/
COPY start.sh /srv/
RUN chmod +x /srv/start.sh

WORKDIR /srv

# Expose the default port
EXPOSE 8000

ENTRYPOINT ["/srv/start.sh"]
