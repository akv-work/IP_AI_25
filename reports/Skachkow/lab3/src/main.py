
import argparse
import os
import sys
from pathlib import Path

import yaml
import requests

ROOT = Path(__file__).resolve().parent


DATA_DIR = ROOT / "cats"
SOURCE_YAML = DATA_DIR / "data.yaml"
DATA_YAML = DATA_DIR / "data_fixed.yaml"

RUNS_DIR = ROOT / "runs"
RUN_NAME = "cats_yolo12s"
BEST_WEIGHTS = (
    RUNS_DIR / "detect" / RUN_NAME / "weights" / "best.pt"
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



def polygon_to_bbox(values):
    """Преобразует сегментационный полигон в bbox для YOLO."""

    cls = int(float(values[0]))
    coords = [float(v) for v in values[1:]]

    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError("Некорректный полигон")

    xs = coords[0::2]
    ys = coords[1::2]

    if not all(0 <= x <= 1 for x in xs):
        raise ValueError("Координаты X вне диапазона [0, 1]")

    if not all(0 <= y <= 1 for y in ys):
        raise ValueError("Координаты Y вне диапазона [0, 1]")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError("Нулевой размер рамки")

    return (
        f"{cls} "
        f"{(x_min + x_max) / 2:.6f} "
        f"{(y_min + y_max) / 2:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def inspect_and_fix_labels():
    """
    Проверяет YOLO-разметку.
    Сегментационные полигоны преобразуются в рамки.
    Удаляются только некорректные строки.
    """

    for split in ("train", "valid", "test"):
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"

        if not img_dir.exists():
            print(f"[data] Сплит {split}: папка images не найдена")
            continue

        images = [
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXT
        ]

        if not lbl_dir.exists():
            print(f"[data] Предупреждение: нет папки {lbl_dir}")
            continue

        n_obj = 0
        n_empty = 0
        n_poly = 0
        n_bad = 0
        per_class = {}

        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"

            if not lbl.exists():
                n_empty += 1
                continue

            try:
                lines = lbl.read_text(encoding="utf-8").splitlines()
            except OSError as e:
                print(f"[data] Не удалось прочитать {lbl}: {e}")
                continue

            new_lines = []

            for line in lines:
                parts = line.split()

                if not parts:
                    continue

                try:
                    # Сегментационная разметка:
                    # class x1 y1 x2 y2 ... xn yn
                    if len(parts) > 5:
                        line = polygon_to_bbox(parts)
                        parts = line.split()
                        n_poly += 1

                    if len(parts) != 5:
                        raise ValueError("Ожидалось 5 значений")

                    cls_float = float(parts[0])
                    if not cls_float.is_integer():
                        raise ValueError("Индекс класса не целый")

                    cls = int(cls_float)
                    xc, yc, w, h = map(float, parts[1:])

                    if cls < 0:
                        raise ValueError("Отрицательный класс")

                    if not all(
                        0 <= v <= 1 for v in (xc, yc)
                    ):
                        raise ValueError("Центр рамки вне [0, 1]")

                    if not (0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("Некорректный размер рамки")

                    if (
                        xc - w / 2 < -1e-6
                        or xc + w / 2 > 1 + 1e-6
                        or yc - h / 2 < -1e-6
                        or yc + h / 2 > 1 + 1e-6
                    ):
                        raise ValueError("Рамка выходит за границы изображения")

                    new_lines.append(
                        f"{cls} {xc:.6f} {yc:.6f} "
                        f"{w:.6f} {h:.6f}"
                    )
                    per_class[cls] = per_class.get(cls, 0) + 1

                except (ValueError, IndexError, OverflowError):
                    n_bad += 1

            # Перезаписываем только существующие файлы разметки.
            lbl.write_text(
                "\n".join(new_lines) + ("\n" if new_lines else ""),
                encoding="utf-8"
            )

            n_obj += len(new_lines)

            if not new_lines:
                n_empty += 1

        print(
            f"[data] {split:5s}: изображений={len(images)}, "
            f"объектов={n_obj}, без объектов={n_empty}, "
            f"полигонов->рамок={n_poly}, "
            f"битых строк={n_bad}, классы={per_class}"
        )


def prepare():
    """Проверяет уже скачанный датасет и создаёт YAML для обучения."""

    if not SOURCE_YAML.exists():
        sys.exit(
            f"Не найден файл {SOURCE_YAML}\n"
            "Проверьте расположение датасета cats."
        )

    for split in ("train", "valid"):
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"

        if not img_dir.exists():
            sys.exit(f"Не найдена папка изображений: {img_dir}")

        if not lbl_dir.exists():
            sys.exit(f"Не найдена папка разметки: {lbl_dir}")

    print(f"[data] Используем датасет: {DATA_DIR}")

    with SOURCE_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    names = cfg.get("names")

    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    else:
        sys.exit("В cats/data.yaml отсутствует корректный параметр names.")

    if not names:
        sys.exit("В cats/data.yaml не указаны классы.")

    nc = cfg.get("nc")
    if nc is not None and int(nc) != len(names):
        sys.exit(
            f"Несоответствие классов: nc={nc}, "
            f"names содержит {len(names)} классов."
        )

    # Важно: индексы классов должны начинаться с нуля.
    if sorted(names) != list(range(len(names))):
        sys.exit(
            f"Индексы классов должны идти от 0 до {len(names) - 1}: "
            f"{names}"
        )

    # До исправления разметки проверяем исходные имена классов.
    print(f"[data] Классы: {names}")

    inspect_and_fix_labels()

    new_cfg = {
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "names": names,
    }

    if (DATA_DIR / "test" / "images").exists():
        new_cfg["test"] = "test/images"

    with DATA_YAML.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            new_cfg,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    print(f"[data] Конфигурация создана: {DATA_YAML}")
    print("[data] Подготовка завершена.")




def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def train(epochs=100, imgsz=640, batch=16):
    from ultralytics import YOLO

    if not DATA_YAML.exists():
        prepare()

    device = 0 if cuda_available() else "cpu"

    print(f"[train] Устройство: {device}")
    print(f"[train] Эпох: {epochs}")
    print(f"[train] Размер изображения: {imgsz}")
    print(f"[train] Batch size: {batch}")

    model = YOLO("yolo12s.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=20,
        project=str(RUNS_DIR / "detect"),
        name=RUN_NAME,
        exist_ok=True,
        device=device,
        workers=4,
        plots=True,
        seed=42,
        save=True,
        verbose=True,
    )

    if not BEST_WEIGHTS.exists():
        sys.exit(
            f"Не найдены лучшие веса: {BEST_WEIGHTS}\n"
            "Проверьте логи обучения."
        )

    print(f"\n[train] Лучшие веса: {BEST_WEIGHTS}")

    # Оцениваем лучшую модель на тестовой выборке,
    # если она присутствует; иначе используем valid.
    with DATA_YAML.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    split = "test" if cfg.get("test") else "val"

    best = YOLO(str(BEST_WEIGHTS))

    metrics = best.val(
        data=str(DATA_YAML),
        split=split,
        imgsz=imgsz,
        device=device,
        project=str(RUNS_DIR / "detect"),
        name=RUN_NAME + "_eval",
        exist_ok=True,
        plots=True,
    )

    print(f"\n========== РЕЗУЛЬТАТЫ ({split}) ==========")
    print(f"Precision     : {metrics.box.mp:.4f}")
    print(f"Recall        : {metrics.box.mr:.4f}")
    print(f"mAP@0.5       : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95  : {metrics.box.map:.4f}")

    for i, name in best.names.items():
        if i < len(metrics.box.maps):
            print(
                f"AP@0.5:0.95 [{name}]: "
                f"{metrics.box.maps[i]:.4f}"
            )

    print(f"\n[train] Графики: {RUNS_DIR / 'detect' / RUN_NAME}")
    print(f"[train] Веса: {BEST_WEIGHTS}")



def predict(sources, conf=0.25, weights=None):
    import cv2
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    weights = Path(weights) if weights else BEST_WEIGHTS

    if not weights.is_absolute():
        weights = (ROOT / weights).resolve()

    if not weights.exists():
        sys.exit(
            f"Веса не найдены: {weights}\n"
            "Сначала выполните этап train."
        )

    if not sources:
        sys.exit("Укажите хотя бы один путь или URL изображения.")

    model = YOLO(str(weights))

    out = ROOT / "predictions"
    tmp = ROOT / "web_images"

    out.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)

    results_imgs = []

    for i, src in enumerate(sources):
        try:
            if src.startswith(("http://", "https://")):
                response = requests.get(
                    src,
                    timeout=30,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                response.raise_for_status()

                path = tmp / f"web_{i}.jpg"
                path.write_bytes(response.content)
            else:
                path = Path(src)

                if not path.is_absolute():
                    path = (Path.cwd() / path).resolve()

            if not path.exists():
                print(f"[predict] Файл не найден: {path}")
                continue

            image = cv2.imread(str(path))

            if image is None:
                print(f"[predict] Не удалось открыть изображение: {path}")
                continue

            result = model.predict(
                source=image,
                conf=conf,
                verbose=False,
            )[0]

            plotted = result.plot()

            output_file = out / f"pred_{i}.jpg"
            if not cv2.imwrite(str(output_file), plotted):
                print(f"[predict] Не удалось сохранить {output_file}")
                continue

            print(f"\nИзображение: {src}")
            print(f"Найдено объектов: {len(result.boxes)}")

            for box in result.boxes:
                cls = int(box.cls.item())
                score = float(box.conf.item())
                print(f"  {model.names[cls]}: {score:.2%}")

            results_imgs.append(
                (str(src), cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
            )

        except (requests.RequestException, OSError, ValueError) as e:
            print(f"[predict] Ошибка для {src}: {e}")

    if not results_imgs:
        sys.exit("Не удалось обработать ни одного изображения.")

    n = len(results_imgs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 5 * rows),
        squeeze=False
    )

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (title, img) in zip(axes.ravel(), results_imgs):
        ax.imshow(img)
        ax.set_title(Path(title).name if not title.startswith("http") else title)

    plt.tight_layout()
    plt.savefig(out / "summary.png", dpi=150, bbox_inches="tight")

    print(f"\n[predict] Результаты сохранены в {out}")
    print(f"[predict] Общая визуализация: {out / 'summary.png'}")

    plt.show()




def main():
    parser = argparse.ArgumentParser(
        description="Обучение и использование YOLO12s на датасете cats."
    )

    parser.add_argument(
        "stage",
        choices=["prepare", "train", "predict", "all"],
        help="Этап: подготовка, обучение, распознавание или оба первых этапа"
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="Пути к фотографиям или URL для этапа predict"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--weights", default=None)

    args = parser.parse_args()

    if args.stage == "prepare":
        prepare()

    elif args.stage == "train":
        train(args.epochs, args.imgsz, args.batch)

    elif args.stage == "predict":
        predict(args.sources, args.conf, args.weights)

    elif args.stage == "all":
        prepare()
        train(args.epochs, args.imgsz, args.batch)


if __name__ == "__main__":
    main()
