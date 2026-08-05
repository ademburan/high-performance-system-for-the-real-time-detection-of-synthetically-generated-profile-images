FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      libopenblas-dev \
      liblapack-dev \
      libjpeg-dev \
      libpng-dev \
      libx11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY worker.py .

# 9100: Prometheus metrics (host'a yayınlanmaz, compose ağı içi)
EXPOSE 9100

CMD ["python", "-u", "worker.py"]
