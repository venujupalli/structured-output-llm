FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip3 install --upgrade pip && \
    pip3 install \
    torch \
    transformers \
    datasets \
    peft \
    bitsandbytes \
    trl \
    accelerate \
    mlflow \
    jsonschema \
    pyyaml

RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
