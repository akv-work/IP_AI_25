import os
import ssl
import urllib.request
from pathlib import Path

import certifi

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
EPOCHS = 10
IMGSZ = 416
BATCH = 4
LIMITS = [300, 40, 200]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
URLS = [
    "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?auto=format&fit=crop&w=960&q=80",
    "https://images.unsplash.com/photo-1511632765486-a01980e01a18?auto=format&fit=crop&w=960&q=80",
    "https://images.unsplash.com/photo-1522202176988-66273c2fd55f?auto=format&fit=crop&w=960&q=80",
]


def api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()
    key_file = ROOT / "roboflow.key"
    if key_file.exists():
        return key_file.read_text(encoding="utf-8").strip()
    config = Path.home() / ".config" / "roboflow" / "config.json"
    if not config.exists():
        return None
    import json

    data = json.loads(config.read_text(encoding="utf-8"))
    workspaces = data.get("workspaces", {})
    current = workspaces.get(data.get("RF_WORKSPACE"), {})
    return current.get("apiKey") or next((item.get("apiKey") for item in workspaces.values() if item.get("apiKey")), None)


def dataset_zip():
    import zipfile

    folders = [ROOT]
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        folders.append(downloads)
    archives = [path for folder in folders for path in folder.glob("*.zip")]
    archives.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    for path in archives:
        try:
            with zipfile.ZipFile(path) as zipped:
                if any(name.endswith("data.yaml") for name in zipped.namelist()):
                    return path
        except zipfile.BadZipFile:
            continue
    return None


def ensure_dataset():
    yamls = [p for p in DATA.rglob("data.yaml")]
    if yamls:
        return yamls[0]
    archive = dataset_zip()
    if archive:
        import zipfile

        DATA.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(DATA)
        return next(DATA.rglob("data.yaml"))
    key = api_key()
    if not key:
        raise SystemExit(
            "Нет датасета People Detection v10.\n"
            "1. Откройте https://app.roboflow.com/settings и скопируйте API key.\n"
            "2. Вставьте ключ одной строкой в файл roboflow.key в папке проекта.\n"
            "Либо на странице датасета скачайте YOLOv11 zip — скрипт возьмёт его из Загрузок."
        )
    from roboflow import Roboflow

    Roboflow(api_key=key).workspace("leo-ueno").project("people-detection-o4rdr").version(10).download(
        "yolov11", location=str(DATA), overwrite=True
    )
    return next(DATA.rglob("data.yaml"))


def prepare(yaml_path: Path) -> Path:
    root = yaml_path.parent
    lines = []
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        key, sep, val = line.partition(":")
        name = key.strip()
        if not (sep and name in {"train", "val", "test"} and val.strip()):
            lines.append(line)
            continue
        rel = val.split("#", 1)[0].strip().strip("\"'")
        fixed = rel[3:] if rel.startswith("../") else rel
        while fixed.startswith("../"):
            fixed = fixed[3:]
        path = (root / fixed).resolve()
        if not path.exists():
            path = (root / rel).resolve()
        lines.append(f"{name}: {path}")
    out = root / "data_local.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def summarize(root: Path):
    counts = {}
    for split in ("train", "valid", "test"):
        images = [p for p in (root / split / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        counts[split] = len(images)
        print(f"{split}: images={len(images)}")
    sample = next((root / "train" / "labels").glob("*.txt"))
    print("пример метки YOLO (class cx cy w h):", " ".join(sample.read_text(encoding="utf-8").split()[:5]))
    return counts


def web_photos():
    folder = ROOT / "samples"
    folder.mkdir(exist_ok=True)
    paths = []
    for i, url in enumerate(URLS, 1):
        dest = folder / f"web_{i}.jpg"
        if not dest.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(req, context=context) as response, dest.open("wb") as file:
                file.write(response.read())
        paths.append(str(dest))
    return paths


def plot_losses():
    import csv
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader((ROOT / "runs" / "train" / "results.csv").open(encoding="utf-8")))
    epochs = [int(row["epoch"]) for row in rows]

    def total(split):
        return [float(row[f"{split}/box_loss"]) + float(row[f"{split}/cls_loss"]) + float(row[f"{split}/dfl_loss"]) for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, total("train"), marker="o", label="train")
    plt.plot(epochs, total("val"), marker="o", label="val")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = ROOT / "runs" / "train" / "loss.png"
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def test_dir(yaml: Path) -> Path:
    for line in yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("test:"):
            return Path(line.split(":", 1)[1].strip())
    raise SystemExit("В data.yaml нет пути test")


def main():
    raw = ensure_dataset()
    yaml = prepare(raw)
    counts = summarize(raw.parent)
    weights = ROOT / "runs" / "train" / "weights" / "best.pt"
    model = YOLO(weights if weights.exists() else "yolo11n.pt")
    if not weights.exists():
        model.train(
            data=str(yaml),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=0,
            amp=False,
            fraction=LIMITS,
            project=str(ROOT / "runs"),
            name="train",
            exist_ok=True,
        )
        model = YOLO(weights)
    metrics = model.val(data=str(yaml), split="test", fraction=LIMITS, device=DEVICE, workers=0, plots=False)
    map50, map5095 = metrics.box.map50, metrics.box.map
    print(f"test mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
    print("график ошибок:", plot_losses())
    model.predict(source=str(test_dir(yaml)), save=True, project=str(ROOT / "runs"), name="test", exist_ok=True, device=DEVICE)
    model.predict(source=web_photos(), save=True, project=str(ROOT / "runs"), name="internet", exist_ok=True, device=DEVICE)
    (ROOT / "отчет.txt").write_text(
        "Лабораторная работа № 3. Вариант 1\n"
        "Детектор: YOLO11n (pretrained yolo11n.pt)\n"
        "Датасет: People Detection v10, https://universe.roboflow.com/leo-ueno/people-detection-o4rdr/dataset/10\n"
        f"Объём: train {counts['train']}, valid {counts['valid']}, test {counts['test']}.\n"
        "Подготовка: экспорт YOLOv11 уже в формате class cx cy w h. "
        "Дополнительная конвертация не нужна, в data.yaml исправлены относительные пути.\n"
        f"Обучение: {EPOCHS} эпох на {LIMITS[0]} кадрах train и {LIMITS[1]} val "
        f"(из {counts['train']} / {counts['valid']}), imgsz={IMGSZ}, batch={BATCH}, device={DEVICE}.\n"
        f"Тест: {LIMITS[2]} кадров из {counts['test']}, mAP50={map50:.4f}, mAP50-95={map5095:.4f}.\n"
        "График ошибки по эпохам: runs/train/loss.png.\n"
        "Детекция на тестовой выборке: runs/test.\n"
        "Визуализация людей на фотографиях из интернета: runs/internet.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
