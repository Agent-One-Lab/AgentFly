FROM ubuntu:24.04

WORKDIR /app

# System deps (Java 21 for pyserini/Lucene)
RUN apt-get update && apt-get install -y \
    openjdk-21-jdk \
    wget \
    pkg-config \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Miniconda + Python 3.10
RUN mkdir -p /root/miniconda3 && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm /root/miniconda3/miniconda.sh
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda install -y python=3.10

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# pyserini's encoder module instantiates openai.OpenAI() at import time and needs a
# key present (WebShop never uses OpenAI — it's Lucene/BM25 only). A dummy key lets the
# import succeed, both during the build-time index step and at server startup.
ENV OPENAI_API_KEY=dummy

# ---- Data (pre-staged in the build context under data/{full,small}; no gdown, which
#      is dead for the canonical Drive IDs). See README for how to stage. ----
ARG DATASET=full
COPY data/${DATASET} /app/data/${DATASET}
COPY data/items_human_ins.json /app/data/

# ---- App code ----
COPY engine.py goal.py normalize.py utils.py webshop_simulator_server.py build_index.py ./
COPY static/ /app/static/
COPY templates/ /app/templates/

# ---- Build the Lucene index at build time (baked into the image; no runtime rebuild).
#      The index must match the catalog: DATASET=small -> ~1,000 docs, full -> ~1.18M. ----
RUN WEBSHOP_DATASET=${DATASET} python build_index.py && \
    python -m pyserini.index.lucene \
      --collection JsonCollection \
      --input /app/resources \
      --index /app/indexes \
      --generator DefaultLuceneDocumentGenerator \
      --threads 16 \
      --storePositions --storeDocvectors --storeRaw && \
    rm -rf /app/resources

# ---- Runtime config: server reads these to pick catalog + goal type ----
#      full -> human goals; small -> synthetic goals (aligned to verl-agent).
ARG HUMAN_GOALS=1
ENV WEBSHOP_DATASET=${DATASET}
ENV WEBSHOP_HUMAN_GOALS=${HUMAN_GOALS}

RUN mkdir -p /tmp /var/tmp /usr/tmp && \
    chmod 1777 /tmp /var/tmp /usr/tmp && \
    chmod -R 777 /app

EXPOSE 3000
COPY start.sh /app
RUN chmod +x /app/start.sh
ENTRYPOINT ["/app/start.sh"]
