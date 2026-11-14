"""
Лабораторная работа №3. Обучение детекторов объектов.
Вариант 6: детектор YOLO11m, датасет «Камень-ножницы-бумага» (Roboflow, rock-paper-scissors-sxsw v14).
"""
import csv
import os

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATA_YAML = os.path.join(BASE_DIR, "rps.yaml")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
RUN_NAME = "yolo11m_rps"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

EPOCHS = 25
IMAGE_SIZE = 640
BATCH_SIZE = 6
WORKERS = 2
SEED = 42
CLASS_NAMES = ["Paper", "Rock", "Scissors"]


def write_data_yaml():
    lines = [
        f"path: {DATASET_DIR.replace(os.sep, '/')}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {CLASS_NAMES}",
    ]
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_results(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {key.strip(): [float(row[key]) for row in rows] for key in rows[0]}


def plot_history(csv_path, path):
    df = read_results(csv_path)
    epochs = df["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, name, title in zip(axes, ["box_loss", "cls_loss", "dfl_loss"],
                               ["Ошибка локализации (box_loss)", "Ошибка классификации (cls_loss)",
                                "Distribution Focal Loss (dfl_loss)"]):
        ax.plot(epochs, df[f"train/{name}"], "o-", markersize=3, label="Train")
        ax.plot(epochs, df[f"val/{name}"], "s-", markersize=3, label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Изменение ошибки при обучении YOLO11m")
    fig.tight_layout()
    fig.savefig(os.path.join(path, "loss.png"), dpi=150)

    fig2, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, df["metrics/mAP50(B)"], "o-", markersize=3, label="mAP@0.5")
    ax.plot(epochs, df["metrics/mAP50-95(B)"], "s-", markersize=3, label="mAP@0.5:0.95")
    ax.plot(epochs, df["metrics/precision(B)"], "--", label="Precision")
    ax.plot(epochs, df["metrics/recall(B)"], "--", label="Recall")
    ax.set_title("Метрики на валидационной выборке")
    ax.set_xlabel("Эпоха")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(path, "metrics.png"), dpi=150)
    return fig, fig2


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    write_data_yaml()
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {torch.cuda.get_device_name(0) if device == 0 else 'cpu'}")

    model = YOLO("yolo11m.pt")
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,
        seed=SEED,
        project=RUNS_DIR,
        name=RUN_NAME,
        exist_ok=True,
        plots=True,
    )

    plot_history(os.path.join(RUNS_DIR, RUN_NAME, "results.csv"), RESULTS_DIR)
    plt.show()


if __name__ == "__main__":
    main()
