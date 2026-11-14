"""
Оценка обученного детектора YOLO11m на тестовой выборке (mAP).
"""
import json
import os

import torch
from ultralytics import YOLO

from train import CLASS_NAMES, DATA_YAML, IMAGE_SIZE, RESULTS_DIR, RUN_NAME, RUNS_DIR, write_data_yaml

WEIGHTS = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")


def evaluate(model, split, device):
    metrics = model.val(data=DATA_YAML, split=split, imgsz=IMAGE_SIZE, batch=4, workers=0, device=device,
                        project=RUNS_DIR, name=f"eval_{split}", exist_ok=True, plots=True)
    box = metrics.box
    result = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": float(box.p[i]),
                "recall": float(box.r[i]),
                "mAP50": float(box.ap50[i]),
                "mAP50-95": float(box.ap[i]),
            }
            for i, c in enumerate(box.ap_class_index)
        },
        "speed_ms": {k: float(v) for k, v in metrics.speed.items()},
    }
    return result


def main():
    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"Веса не найдены: {WEIGHTS}. Сначала запустите train.py.")
    write_data_yaml()
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(WEIGHTS)

    results = {split: evaluate(model, split, device) for split in ("val", "test")}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for split, name in (("val", "Валидационная"), ("test", "Тестовая")):
        r = results[split]
        print(f"\n{name} выборка: Precision={r['precision']:.3f}  Recall={r['recall']:.3f}  "
              f"mAP@0.5={r['mAP50']:.3f}  mAP@0.5:0.95={r['mAP50-95']:.3f}")
        print(f"{'Класс':<10}{'Precision':>11}{'Recall':>9}{'mAP@0.5':>10}{'mAP@0.5:0.95':>14}")
        for cls, m in r["per_class"].items():
            print(f"{cls:<10}{m['precision']:>11.3f}{m['recall']:>9.3f}{m['mAP50']:>10.3f}{m['mAP50-95']:>14.3f}")


if __name__ == "__main__":
    main()
