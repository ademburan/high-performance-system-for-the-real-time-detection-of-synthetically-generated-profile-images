"""
benchmark.py - GANEyeDistance Karşılaştırma Benchmarkı

Orijinal paper metodunu tek thread olarak çalıştırır, süreyi ölçer.
Amacı: bizim asenkron sistemimizin hız avantajını ortaya koymak.

Dlib 5-nokta model nokta sırası:
  0, 1 -> sağ göz
  2, 3 -> sol göz
  4    -> burun

Kullanım:
  py -3.12 scripts\\benchmark.py --src=bot_profiles --out=benchmark_bot.csv
  py -3.12 scripts\\benchmark.py --src=real_profiles --out=benchmark_real.csv
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import dlib
import numpy as np
from PIL import Image, UnidentifiedImageError

# Model dosyası yolu
MODELS_DIR = (
    r"C:\Users\ademb\AppData\Local\Packages"
    r"\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python312\site-packages"
    r"\face_recognition_models\models"
)
PREDICTOR_5_PATH = os.path.join(MODELS_DIR, "shape_predictor_5_face_landmarks.dat")

if not os.path.exists(PREDICTOR_5_PATH):
    print(f"HATA: Model dosyası bulunamadı: {PREDICTOR_5_PATH}")
    sys.exit(1)

# Modelleri bir kere yükle
DETECTOR  = dlib.get_frontal_face_detector()
PREDICTOR = dlib.shape_predictor(PREDICTOR_5_PATH)

# Paper'ın orijinal ground truth koordinatları
LEFT_EYE_GT  = (0.3808205078125, 0.4770548828125)
RIGHT_EYE_GT = (0.6152169921875, 0.4771314453125)
THRESHOLD    = 0.02

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def calculate_ganeye_distance(image_path: str) -> tuple:
    """
    Paper'ın orijinal GANEyeDistance metodu.

    Dlib 5-nokta model nokta sırası:
      0, 1 -> sağ göz (right eye)
      2, 3 -> sol göz (left eye)
      4    -> burun
    """
    try:
        pil_image = Image.open(image_path).convert("RGB")
        image_np  = np.array(pil_image)
        size_x, size_y = pil_image.size

        faces = DETECTOR(image_np, 1)

        if len(faces) == 0:
            return 1.0, "no_face_detected"
        elif len(faces) > 1:
            return 1.0, "multiple_faces_detected"

        shape  = PREDICTOR(image_np, faces[0])
        points = [(shape.part(i).x, shape.part(i).y) for i in range(5)]

        # DÜZELTME: 0,1 sağ göz — 2,3 sol göz
        re_x = (points[0][0] + points[1][0]) / 2
        re_y = (points[0][1] + points[1][1]) / 2
        le_x = (points[2][0] + points[3][0]) / 2
        le_y = (points[2][1] + points[3][1]) / 2

        # Normalize et
        le_x_norm = le_x / size_x
        le_y_norm = le_y / size_y
        re_x_norm = re_x / size_x
        re_y_norm = re_y / size_y

        le_x_gt, le_y_gt = LEFT_EYE_GT
        re_x_gt, re_y_gt = RIGHT_EYE_GT

        # GANEyeDistance formülü (paper ile birebir)
        distance = math.sqrt(
            (le_x_norm - le_x_gt) ** 2 + (le_y_norm - le_y_gt) ** 2
        ) + math.sqrt(
            (re_x_norm - re_x_gt) ** 2 + (re_y_norm - re_y_gt) ** 2
        )

        distance_norm = distance / (2 * math.sqrt(2))
        return round(distance_norm, 8), ""

    except (UnidentifiedImageError, OSError) as e:
        return 1.0, f"invalid_image: {e}"
    except Exception as e:
        return 1.0, f"exception: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",   required=True, help="Fotoğraf klasörü")
    ap.add_argument("--out",   required=True, help="Çıktı CSV dosyası")
    ap.add_argument("--limit", type=int, default=0,
                    help="İşlenecek maksimum fotoğraf sayısı (0=hepsi)")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"HATA: Klasör bulunamadı: {src}")
        sys.exit(1)

    images = sorted([
        p for p in src.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if args.limit > 0:
        images = images[:args.limit]

    total = len(images)
    print(f"Toplam {total} fotoğraf: {src}")
    print(f"Eşik: {THRESHOLD}")
    print(f"Çıktı: {args.out}")
    print("-" * 50)

    results     = []
    bot_count   = 0
    real_count  = 0
    error_count = 0
    start_total = time.time()

    for i, img_path in enumerate(images, 1):
        t0 = time.time()
        score, error = calculate_ganeye_distance(str(img_path))
        elapsed_ms = int((time.time() - t0) * 1000)

        if error:
            label = "unknown"
            error_count += 1
        elif score < THRESHOLD:
            label = "bot"
            bot_count += 1
        else:
            label = "real"
            real_count += 1

        results.append({
            "filename":      img_path.name,
            "score":         score,
            "label":         label,
            "processing_ms": elapsed_ms,
            "error":         error,
        })

        if i % 100 == 0 or i == total:
            elapsed = time.time() - start_total
            hiz = i / elapsed if elapsed > 0 else 0
            print(f"[{i}/{total}] bot={bot_count} real={real_count} "
                  f"error={error_count} hiz={hiz:.1f} foto/sn")

    total_elapsed = time.time() - start_total

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "score", "label", "processing_ms", "error"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 50)
    print(f"TAMAMLANDI")
    print(f"Toplam fotoğraf : {total}")
    print(f"Bot             : {bot_count} ({bot_count/total*100:.1f}%)")
    print(f"Real            : {real_count} ({real_count/total*100:.1f}%)")
    print(f"Hata            : {error_count} ({error_count/total*100:.1f}%)")
    print(f"Toplam süre     : {total_elapsed:.1f} saniye")
    print(f"Ortalama hız    : {total/total_elapsed:.1f} fotoğraf/saniye")
    print(f"Fotoğraf başına : {total_elapsed/total*1000:.0f} ms")
    print(f"Çıktı           : {args.out}")
    print("=" * 50)


if __name__ == "__main__":
    main()
