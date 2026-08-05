# GANEye Bot Detection System

> A scalable, asynchronous microservices pipeline for detecting StyleGAN-generated fake profile images on social networks using the GANEyeDistance algorithm.

[![Go](https://img.shields.io/badge/Go-1.22-blue?logo=go)](https://go.dev/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org/)
[![RabbitMQ](https://img.shields.io/badge/RabbitMQ-3.13-orange?logo=rabbitmq)](https://www.rabbitmq.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

**GANEye Bot Detection System** implements the [GANEyeDistance algorithm](https://arxiv.org/abs/2401.02627) (Yang, Singh & Menczer, 2024) within a production-ready, horizontally scalable distributed architecture. The algorithm detects GAN-generated faces by measuring the deviation of detected eye positions from the fixed normalized coordinates inherent to StyleGAN's FFHQ training alignment — a structural artifact that causes synthetic faces to cluster tightly around a known spatial signature.

This project was developed as a **bachelor's thesis**, demonstrating that a research-grade detection algorithm can be operationalized at scale without modifying its core logic.

### Key Results

| Metric | Value |
|---|---|
| Recall on TwitterGAN (1,420 verified bots) | **99.4%** |
| Recall reported in source paper | 99.5% |
| Throughput (4 workers vs. single-thread) | **13.84× faster** |
| Dataset processed | 28,599 images (TwitterGAN + RandomTwitter) |

---

## Architecture

```
  Client / Traffic Generator
          │
          │  POST /detect  (multipart/form-data)
          ▼
  ┌─────────────────────┐
  │   Go API Gateway    │  :8080 — UUID, semaphore, publish timeout
  └──────────┬──────────┘
             │  tasks queue (durable, persistent)
             ▼
  ┌──────────────────────────────┐
  │         RabbitMQ 3.13        │  :5672 / :15672 (management UI)
  └──────────────────────────────┘
             │  consume (prefetch=4)
             ▼
  ┌─────────────────────┐  × N replicas
  │   Python Worker     │  dlib · face_recognition · GANEyeDistance
  └──────────┬──────────┘
             │  results queue
             ▼
  ┌─────────────────────┐        ┌──────────────┐
  │   Go Aggregator     │──────▶ │  PostgreSQL  │  :5432
  │   :8081             │        │  WAL mode    │
  └─────────────────────┘        └──────────────┘
             │
  ┌──────────▼──────────┐
  │  Prometheus + Grafana│  :9090 / :3000
  └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Ingestion | Go 1.22, `net/http`, `amqp091-go`, `uuid` |
| Message Broker | RabbitMQ 3.13 (AMQP 0-9-1) |
| Inference | Python 3.11, dlib, face_recognition, NumPy, Pillow |
| Persistence | PostgreSQL 16 (pure-Go `pgx/v5`, WAL + upsert) |
| Observability | Prometheus v2.53, Grafana 11.1 |
| Orchestration | Docker, Docker Compose v2 |
| Load Testing | Go (token-bucket rate limiter, `x/time/rate`) |

---

## Project Structure

```
ganeye-bot-detection/
├── docker-compose.yml          # Full service orchestration
├── gateway/                    # Go API Gateway
│   ├── main.go
│   ├── go.mod
│   └── Dockerfile
├── worker/                     # Python inference worker
│   ├── worker.py
│   ├── requirements.txt
│   └── Dockerfile
├── aggregator/                 # Go result aggregator + HTTP API
│   ├── main.go
│   ├── go.mod
│   └── Dockerfile
├── loadgen/                    # Go traffic generator
│   ├── main.go
│   └── go.mod
├── scripts/
│   ├── evaluate.py             # Accuracy evaluation (confusion matrix, recall, F1)
│   └── benchmark.py            # Single-threaded reference benchmark
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── provisioning/
│           ├── datasources/
│           └── dashboards/
├── rabbitmq/
│   └── enabled_plugins
└── data/                       # PostgreSQL bind-mount (create manually)
```

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose v2)
- [Go 1.22+](https://go.dev/dl/) — for the traffic generator (runs on host)
- [Python 3.11+](https://www.python.org/downloads/) — for evaluation scripts

---

## Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ganeye-bot-detection.git
cd ganeye-bot-detection
```

**2. Create the data directory:**
```bash
mkdir data
```

**3. Build and start all services:**
```bash
docker compose build    # First build ~5–10 min (dlib compilation)
docker compose up -d
```

**4. Verify services are running:**
```bash
docker compose ps
curl http://localhost:8080/health   # Gateway
curl http://localhost:8081/health   # Aggregator
```

**5. Open monitoring interfaces:**

| Service | URL | Credentials |
|---|---|---|
| RabbitMQ Management | http://localhost:15672 | guest / guest |
| Prometheus | http://localhost:9090 | — |
| Grafana Dashboard | http://localhost:3000 | admin / admin |

---

## Usage

### Submit a Single Image

```bash
# Linux / macOS
curl -X POST http://localhost:8080/detect \
     -F "image=@/path/to/profile.jpg"

# Response
{"uuid":"550e8400-...","status":"queued","bytes":24576}
```

### Retrieve a Result

```bash
curl http://localhost:8081/results/550e8400-...

# Response
{
  "uuid": "550e8400-...",
  "label": "bot",
  "score": 0.018742,
  "filename": "profile.jpg"
}
```

### View Statistics

```bash
curl http://localhost:8081/stats

# Response
{"total":1420,"bot":1411,"real":8,"unknown":1}
```

---

## Load Testing

The traffic generator reads images from a directory or `.tar.gz` archive and submits them to the gateway at a controlled rate.

```bash
cd loadgen
go mod tidy
go run . \
    -src=../TwitterGAN_profiles.tar.gz \
    -rps=50 \
    -workers=32 \
    -mapout=../run_mapping.csv
```

| Flag | Default | Description |
|---|---|---|
| `-src` | — | Source `.tar.gz` file or image directory |
| `-rps` | 50 | Target requests per second (token bucket) |
| `-workers` | 32 | Parallel HTTP goroutines |
| `-duration` | 0 | Test duration (0 = send all images once) |
| `-mapout` | — | Output CSV: `filename,uuid,http_status` |

---

## Horizontal Scaling

Scale worker replicas at runtime — no code changes required:

```bash
docker compose up -d --scale worker=4
```

### Measured Throughput

| Configuration | Dataset | Processing Time | Throughput | Speedup |
|---|---|---|---|---|
| Single-thread (baseline) | 1,420 images | 117.2 s | 12.1 img/s | 1.0× |
| 2 workers | 1,420 images | 48 s | 29.6 img/s | 2.44× |
| 4 workers | 1,420 images | ~7 s | ~203 img/s | ~16.7× |
| Single-thread (baseline) | 27,179 images | 4,111 s | 6.6 img/s | 1.0× |
| 2 workers | 27,179 images | 622 s | 43.7 img/s | 6.61× |
| 4 workers | 27,179 images | ~297 s | ~91.5 img/s | **13.84×** |

---

## Accuracy Evaluation

After running a load test with `-mapout`, evaluate detection accuracy against ground-truth labels:

```bash
pip install psycopg2-binary

python3 scripts/evaluate.py \
    --pg "postgresql://ganeye:ganeye@localhost:5432/ganeye" \
    --mapping run_mapping.csv \
    --truth TwitterGAN_id_label_mapping.csv \
    --out evaluation_report.json
```

**Results on TwitterGAN dataset (1,420 verified GAN-generated images):**

| Metric | Value |
|---|---|
| True Positives (bot → bot) | 1,411 |
| False Negatives (bot → real) | 8 |
| **Recall** | **99.4%** |
| Paper reported recall | 99.5% |

---

## Algorithm: GANEyeDistance

StyleGAN is trained on the FFHQ dataset, which applies a canonical alignment that places eyes at near-fixed normalized coordinates in every training image. Generated faces inherit this spatial regularity; authentic photographs do not.

The detection metric for an image is defined as:

$$\mathcal{G}(\vec{L}, \vec{R}) = \frac{\|\vec{L} - \vec{L}_{GAN}\| + \|\vec{R} - \vec{R}_{GAN}\|}{2\sqrt{2}}$$

where:
- $\vec{L}$, $\vec{R}$ — normalized detected eye coordinates
- $\vec{L}_{GAN} = (0.3808, 0.4770)$, $\vec{R}_{GAN} = (0.6152, 0.4771)$ — FFHQ ground-truth coordinates
- A score below **τ = 0.02** is classified as `bot`

> Source: Yang, Singh & Menczer, *"Characteristics and Prevalence of Fake Social Media Profiles with AI-generated Faces"*, arXiv:2401.02627, 2024.

---

## Dataset

This project was evaluated on the [TwitterGAN dataset](https://zenodo.org/doi/10.5281/zenodo.10436888):

- `TwitterGAN_profiles.tar.gz` — 1,420 GAN-generated bot profile images
- `TwitterGAN_id_label_mapping.csv` — Ground-truth labels

Place both files in the project root before running the traffic generator.

---

## Environment Variables

| Service | Variable | Default | Description |
|---|---|---|---|
| gateway | `MAX_UPLOAD_MB` | 10 | Maximum upload size |
| gateway | `PUBLISH_TIMEOUT_MS` | 2000 | Broker publish timeout |
| gateway | `MAX_CONCURRENT_REQUESTS` | 256 | Concurrency semaphore cap |
| worker | `BOT_THRESHOLD` | 0.02 | GANEyeDistance classification threshold |
| worker | `GT_LEFT_EYE_X/Y` | 0.3808 / 0.4770 | Left eye ground-truth coordinates |
| worker | `GT_RIGHT_EYE_X/Y` | 0.6152 / 0.4771 | Right eye ground-truth coordinates |
| worker | `PREFETCH_COUNT` | 4 | AMQP prefetch per worker |
| aggregator | `DB_PATH` | /data/results.db | Database path |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| CGO disabled (`CGO_ENABLED=0`) | Eliminates dlib–Go runtime instability; pure-Go SQLite/PostgreSQL drivers used throughout |
| Python worker isolation | dlib is not thread-safe; per-process isolation enables true parallelism via horizontal scaling |
| Durable queues + persistent delivery | Broker restarts do not lose queued or in-flight tasks |
| PostgreSQL over SQLite | Multi-writer MVCC required for concurrent worker writes at scale |
| Idempotent upsert | Handles RabbitMQ at-least-once delivery without deduplication logic |
| Auto-restarting consumer loop | Broker disconnection degrades to a brief pause, not a silent permanent halt |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this work, please cite the original GANEyeDistance paper:

```bibtex
@article{yang2024characteristics,
  title   = {Characteristics and Prevalence of Fake Social Media Profiles with AI-generated Faces},
  author  = {Yang, Kai-Cheng and Singh, Pranav and Menczer, Filippo},
  journal = {arXiv preprint arXiv:2401.02627},
  year    = {2024}
}
```
