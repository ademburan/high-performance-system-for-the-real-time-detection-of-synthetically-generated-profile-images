"""
evaluate.py (v2) - GANEye Detection doğruluk değerlendiricisi.



Gereksinim: pip install psycopg2-binary

Kullanım:
  python3 scripts/evaluate.py \
      --pg "postgresql://ganeye:ganeye@localhost:5432/ganeye" \
      --mapping run_mapping.csv \
      --truth TwitterGAN_id_label_mapping.csv \
      --out evaluation_report.json
"""
import argparse
import csv
import json
import sys
from collections import Counter

import psycopg2


def norm_label(v: str) -> str:
    v = str(v).strip().lower()
    if v in ("bot", "fake", "gan", "1", "true"):
        return "bot"
    if v in ("real", "human", "0", "false"):
        return "real"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg", default="postgresql://ganeye:ganeye@localhost:5432/ganeye",
                    help="PostgreSQL bağlantı string'i")
    ap.add_argument("--mapping", required=True, help="loadgen -mapout CSV (filename,uuid,http_status)")
    ap.add_argument("--truth", required=True, help="TwitterGAN_id_label_mapping.csv")
    ap.add_argument("--out", default="evaluation_report.json", help="rapor dosyası")
    args = ap.parse_args()

    # filename -> uuid (loadgen çıktısından)
    fn_to_uuid = {}
    with open(args.mapping, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("uuid"):
                fn_to_uuid[row["filename"]] = row["uuid"]

    # filename -> ground truth label
    truth = {}
    with open(args.truth, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fn_field = "filename" if "filename" in reader.fieldnames else reader.fieldnames[0]
        lb_field = "label" if "label" in reader.fieldnames else reader.fieldnames[-1]
        for row in reader:
            truth[row[fn_field].strip()] = norm_label(row[lb_field])

    # uuid -> tahmin (PostgreSQL'den)
    conn = psycopg2.connect(args.pg)
    cur = conn.cursor()
    cur.execute("SELECT uuid, label, score, error FROM results")
    pred = {u: {"label": lb, "score": sc, "error": er} for u, lb, sc, er in cur.fetchall()}
    cur.close()
    conn.close()

    tp = fp = tn = false_neg = 0  # pozitif sınıf = bot
    errors = Counter()
    missing_pred = 0
    missing_truth = 0

    for fname, true_lb in truth.items():
        uu = fn_to_uuid.get(fname)
        if not uu or uu not in pred:
            missing_pred += 1
            continue
        p = pred[uu]
        if p["error"]:
            errors[p["error"]] += 1
            continue
        pl = p["label"]
        if true_lb == "bot" and pl == "bot":
            tp += 1
        elif true_lb == "real" and pl == "bot":
            fp += 1
        elif true_lb == "real" and pl == "real":
            tn += 1
        elif true_lb == "bot" and pl == "real":
            false_neg += 1

    for fname in fn_to_uuid:
        if fname not in truth:
            missing_truth += 1

    total = tp + fp + tn + false_neg
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + false_neg) if (tp + false_neg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    report = {
        "confusion_matrix": {
            "true_positive_bot": tp,
            "false_positive_bot": fp,
            "true_negative_real": tn,
            "false_negative_real": false_neg,
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_bot": round(precision, 4),
            "recall_bot": round(recall, 4),
            "f1_bot": round(f1, 4),
        },
        "counts": {
            "evaluated": total,
            "missing_prediction": missing_pred,
            "missing_truth_label": missing_truth,
            "worker_errors": dict(errors),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
