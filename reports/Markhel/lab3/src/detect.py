"""
Визуализация работы детектора YOLO11m: обнаружение жестов «камень», «ножницы», «бумага»
на произвольных фотографиях.

Запуск:
    python detect.py                     - откроется окно выбора одного или нескольких изображений
    python detect.py фото1.jpg фото2.jpg - обработать указанные файлы
    python detect.py папка               - обработать все изображения в папке
"""
import os
import sys

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

from train import RESULTS_DIR, RUN_NAME, RUNS_DIR

WEIGHTS = os.path.join(RUNS_DIR, RUN_NAME, "weights", "best.pt")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "detections")
CONFIDENCE = 0.5
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_NAMES_RU = {"Paper": "бумага", "Rock": "камень", "Scissors": "ножницы"}


def choose_files():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return []
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Выберите изображения",
        filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Все файлы", "*.*")],
    )
    root.destroy()
    return list(paths)


def collect_paths(args):
    paths = []
    for arg in args:
        if os.path.isdir(arg):
            paths += [os.path.join(arg, f) for f in sorted(os.listdir(arg)) if f.lower().endswith(IMAGE_EXTENSIONS)]
        else:
            paths.append(arg)
    return paths


def main():
    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"Веса не найдены: {WEIGHTS}. Сначала запустите train.py.")
    paths = collect_paths(sys.argv[1:]) if len(sys.argv) > 1 else choose_files()
    if not paths:
        raise SystemExit("Изображения не выбраны.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO(WEIGHTS)
    device = 0 if torch.cuda.is_available() else "cpu"

    for path in paths:
        result = model.predict(path, conf=CONFIDENCE, device=device, verbose=False)[0]
        name = os.path.basename(path)
        print(f"\n{name}: найдено объектов {len(result.boxes)}")
        for box in result.boxes:
            cls = result.names[int(box.cls)]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f"  {cls:<9} ({CLASS_NAMES_RU[cls]:<7}) уверенность {float(box.conf):.2f}  "
                  f"рамка [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        annotated = result.plot(line_width=3)[:, :, ::-1]
        save_path = os.path.join(OUTPUT_DIR, os.path.splitext(name)[0] + "_det.jpg")
        plt.imsave(save_path, annotated)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(annotated)
        ax.set_title(f"{name}: найдено объектов {len(result.boxes)}")
        ax.axis("off")
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
