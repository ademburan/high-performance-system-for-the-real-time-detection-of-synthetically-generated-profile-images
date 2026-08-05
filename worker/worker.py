"""
GANEye Detection Worker (v3)
----------------------------

Paper'ın orijinal GANEyeDistance implementasyonu birebir kullanılıyor:
  - face_recognition kütüphanesi, 5-nokta modeli (model="small")
  - Ground truth: left=(0.3808205078125, 0.4770548828125)
                  right=(0.6152169921875, 0.4771314453125)
  - Normalizasyon: distance / (2 * sqrt(2))
  - Eşik: 0.02 (paper'da %99.5 recall)

Kaynak: https://github.com/... (TwitterGAN paper)
"""

import base64
import io
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pika
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Histogram, start_http_server

import face_recognition

logging.basicConfig(
    level=logging.INFO,
    format="[worker] %(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("ganeye")

# ---------------------------------------------------------------------------
# Prometheus metrikleri
# ---------------------------------------------------------------------------
M_PROCESSED = Counter(
    "ganeye_worker_processed_total",
    "İşlenen görsel sayısı, sınıf etiketi ve hata tipine göre.",
    ["label", "error"],
)
M_PROC_TIME = Histogram(
    "ganeye_worker_processing_seconds",
    "Görsel başına işleme süresi.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)


@dataclass
class Config:
    rabbit_url: str
    task_queue: str
    result_queue: str
    prefetch: int
    metrics_port: int
    # Paper'ın orijinal ground truth koordinatları
    left_eye_gt: Tuple[float, float]
    right_eye_gt: Tuple[float, float]
    threshold: float

    @classmethod
    def load(cls) -> "Config":
        return cls(
            rabbit_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
            task_queue=os.getenv("TASK_QUEUE", "tasks"),
            result_queue=os.getenv("RESULT_QUEUE", "results"),
            prefetch=int(os.getenv("PREFETCH_COUNT", "4")),
            metrics_port=int(os.getenv("METRICS_PORT", "9100")),
            # Paper'ın orijinal değerleri - env ile override edilebilir
            left_eye_gt=(
                float(os.getenv("GT_LEFT_EYE_X", "0.3808205078125")),
                float(os.getenv("GT_LEFT_EYE_Y", "0.4770548828125")),
            ),
            right_eye_gt=(
                float(os.getenv("GT_RIGHT_EYE_X", "0.6152169921875")),
                float(os.getenv("GT_RIGHT_EYE_Y", "0.4771314453125")),
            ),
            threshold=float(os.getenv("BOT_THRESHOLD", "0.02")),
        )


def calculate_ganeye_distance(image_np: np.ndarray, pil_image: Image.Image, cfg: Config) -> float:
    """
    Paper'ın orijinal GANEyeDistance hesaplama metodu.

    - face_recognition 5-nokta modeli (model="small") kullanır
    - Her göz için 2 nokta döner, orta noktası alınır
    - Normalize edilmiş Öklid uzaklığı: distance / (2 * sqrt(2))
    - Yüz bulunamazsa veya birden fazla yüz varsa 1.0 döner
    """
    size_x, size_y = pil_image.size

    # 5-nokta modeli: daha hızlı, paper'ın kullandığı model
    face_landmarks_list = face_recognition.face_landmarks(image_np, model="small")

    if len(face_landmarks_list) == 0:
        return None, "no_face_detected"
    elif len(face_landmarks_list) > 1:
        return None, "multiple_faces_detected"

    face_landmarks = face_landmarks_list[0]

    # Her göz için 2 nokta döner, orta noktasını al
    (le_x1, le_y1), (le_x2, le_y2) = face_landmarks["left_eye"]
    (re_x1, re_y1), (re_x2, re_y2) = face_landmarks["right_eye"]

    le_x = (le_x1 + le_x2) / 2
    le_y = (le_y1 + le_y2) / 2
    re_x = (re_x1 + re_x2) / 2
    re_y = (re_y1 + re_y2) / 2

    # Görüntü boyutuna göre normalize et
    le_x_norm = le_x / size_x
    le_y_norm = le_y / size_y
    re_x_norm = re_x / size_x
    re_y_norm = re_y / size_y

    # Ground truth koordinatları
    le_x_gt, le_y_gt = cfg.left_eye_gt
    re_x_gt, re_y_gt = cfg.right_eye_gt

    # GANEyeDistance formülü (paper ile birebir aynı)
    distance = math.sqrt(
        (le_x_norm - le_x_gt) ** 2 + (le_y_norm - le_y_gt) ** 2
    ) + math.sqrt(
        (re_x_norm - re_x_gt) ** 2 + (re_y_norm - re_y_gt) ** 2
    )

    # Normalizasyon: 2 * sqrt(2) (1x1 karede iki göz arası maksimum mesafe)
    distance_norm = distance / (2 * math.sqrt(2))

    return distance_norm, None


def process_image_bytes(img_bytes: bytes, cfg: Config) -> dict:
    """
    Ham görsel byte'larından tam sonuç sözlüğü üretir.
    Hiçbir durumda exception fırlatmaz.
    """
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        return {"error": "invalid_image", "detail": str(e), "score": 1.0, "label": "unknown"}

    image_np = np.array(pil)

    try:
        score, error = calculate_ganeye_distance(image_np, pil, cfg)
    except Exception as e:
        return {"error": "processing_exception", "detail": str(e), "score": 1.0, "label": "unknown"}

    if error == "no_face_detected":
        return {"error": "no_face_detected", "score": 1.0, "label": "unknown", "face_count": 0}
    if error == "multiple_faces_detected":
        return {"error": "multiple_faces_detected", "score": 1.0, "label": "unknown", "face_count": "multiple"}

    # Eşik: paper'da 0.02 → %99.5 recall
    label = "bot" if score < cfg.threshold else "real"

    return {
        "error": None,
        "score": round(score, 8),
        "label": label,
        "face_count": 1,
        "image_w": pil.size[0],
        "image_h": pil.size[1],
        "threshold": cfg.threshold,
    }


def connect_rabbit(url: str, retries: int = 30, delay: float = 1.0) -> pika.BlockingConnection:
    last_err: Optional[Exception] = None
    params = pika.URLParameters(url)
    params.heartbeat = 120
    params.blocked_connection_timeout = 60
    for i in range(retries):
        try:
            return pika.BlockingConnection(params)
        except pika.exceptions.AMQPConnectionError as e:
            last_err = e
            log.warning("RabbitMQ bağlantı denemesi %d/%d: %s", i + 1, retries, e)
            time.sleep(delay)
    raise RuntimeError(f"RabbitMQ bağlanamadı: {last_err}")


def main() -> None:
    cfg = Config.load()
    log.info("GANEyeDistance worker başlatılıyor")
    log.info("GT left=%.10f,%.10f right=%.10f,%.10f threshold=%.4f",
             cfg.left_eye_gt[0], cfg.left_eye_gt[1],
             cfg.right_eye_gt[0], cfg.right_eye_gt[1],
             cfg.threshold)

    start_http_server(cfg.metrics_port)
    log.info("Prometheus metrics: :%d", cfg.metrics_port)

    conn = connect_rabbit(cfg.rabbit_url)
    channel = conn.channel()
    channel.queue_declare(queue=cfg.task_queue, durable=True)
    channel.queue_declare(queue=cfg.result_queue, durable=True)
    channel.basic_qos(prefetch_count=cfg.prefetch)

    def on_message(ch, method, properties, body: bytes) -> None:
        t0 = time.time()
        task_uuid = "?"
        try:
            task = json.loads(body)
            task_uuid = task.get("uuid", "?")
            img_b64 = task.get("image_b64", "")
            img_bytes = base64.b64decode(img_b64, validate=False) if img_b64 else b""

            if not img_bytes:
                result = {"uuid": task_uuid, "error": "empty_payload", "score": 1.0, "label": "unknown"}
            else:
                r = process_image_bytes(img_bytes, cfg)
                r["uuid"] = task_uuid
                r["filename"] = task.get("filename", "")
                r["processing_ms"] = int((time.time() - t0) * 1000)
                result = r

        except json.JSONDecodeError:
            result = {"uuid": task_uuid, "error": "invalid_json", "score": 1.0, "label": "unknown"}
        except Exception as e:
            log.exception("Beklenmedik hata")
            result = {"uuid": task_uuid, "error": "processing_exception",
                      "detail": str(e), "score": 1.0, "label": "unknown"}

        try:
            ch.basic_publish(
                exchange="",
                routing_key=cfg.result_queue,
                body=json.dumps(result).encode("utf-8"),
                properties=pika.BasicProperties(
                    content_type="application/json",
                    delivery_mode=2,
                ),
            )
        except Exception as e:
            log.error("Publish başarısız uuid=%s err=%s", task_uuid, e)
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
            return

        ch.basic_ack(delivery_tag=method.delivery_tag)

        elapsed = time.time() - t0
        M_PROC_TIME.observe(elapsed)
        M_PROCESSED.labels(
            label=result.get("label", "unknown"),
            error=result.get("error") or "",
        ).inc()

        log.info("uuid=%s score=%s label=%s err=%s ms=%d",
                 task_uuid, result.get("score"), result.get("label"),
                 result.get("error"), int(elapsed * 1000))

    channel.basic_consume(queue=cfg.task_queue, on_message_callback=on_message, auto_ack=False)

    def _graceful(*_):
        log.info("Kapanış sinyali alındı")
        try:
            channel.stop_consuming()
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _graceful)
    signal.signal(signal.SIGINT, _graceful)

    log.info("Kuyruk dinleniyor: %s", cfg.task_queue)
    try:
        channel.start_consuming()
    finally:
        try:
            conn.close()
        except Exception:
            pass
        log.info("Worker durduruldu")


if __name__ == "__main__":
    main()
