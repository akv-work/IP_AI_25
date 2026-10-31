import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from ultralytics import YOLO

MODEL_NAME = "yolo12m.pt"
DATASET_DIR = Path("watermeter_dataset")
DATA_YAML = DATASET_DIR / "data_fixed.yaml"
RUN_DIR = Path("runs") / "yolo12m_watermeter"
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

RF_WORKSPACE = "koer3741-gmail-com"
RF_PROJECT = "watermeteramrv2"
RF_VERSION = 1


def download(api_key):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    project.version(RF_VERSION).download("yolov11", location=str(DATASET_DIR))
    print(f"Датасет загружен в {DATASET_DIR.resolve()}")


def prepare_data():
    src_yaml = DATASET_DIR / "data.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(
            f"Не найден {src_yaml}. Скачайте датасет командой download "
            f"или распакуйте архив YOLO с Roboflow в папку {DATASET_DIR}"
        )
    with open(src_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = DATASET_DIR.resolve()
    splits = {}
    for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if (root / folder / "images").exists():
            splits[key] = f"{folder}/images"
    if "train" not in splits or "val" not in splits:
        raise FileNotFoundError("В датасете должны быть папки train/images и valid/images")

    fixed = {"path": str(root), **splits, "nc": cfg["nc"], "names": cfg["names"]}
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(fixed, f, allow_unicode=True, sort_keys=False)

    print("Классы:", cfg["names"])
    for key, rel in splits.items():
        n = len([p for p in (root / rel).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        print(f"{key}: {n} изображений")
    return "test" in splits


def plot_results():
    csv_path = RUN_DIR / "results.csv"
    with open(csv_path, encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    col = lambda name: [float(r[name]) for r in rows]
    epochs = col("epoch")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for loss in ("box_loss", "cls_loss", "dfl_loss"):
        ax[0].plot(epochs, col(f"train/{loss}"), label=loss)
        ax[1].plot(epochs, col(f"val/{loss}"), label=loss)
    ax[0].set_title("Ошибка на обучающей выборке")
    ax[1].set_title("Ошибка на валидационной выборке")
    ax[2].plot(epochs, col("metrics/mAP50(B)"), label="mAP@0.5")
    ax[2].plot(epochs, col("metrics/mAP50-95(B)"), label="mAP@0.5:0.95")
    ax[2].set_title("mAP на валидационной выборке")
    for a in ax:
        a.set_xlabel("Эпоха")
        a.grid(True)
        a.legend()
    ax[0].set_ylabel("Loss")
    ax[2].set_ylabel("mAP")
    plt.tight_layout()
    plt.savefig("loss_plot_yolo12m.png", dpi=150)
    plt.show()


def evaluate(has_test=None):
    if has_test is None:
        has_test = prepare_data()
    split = "test" if has_test else "val"
    model = YOLO(str(BEST_WEIGHTS))
    metrics = model.val(data=str(DATA_YAML), split=split, imgsz=640, device=DEVICE,
                        project=str(RUN_DIR.parent), name=f"{RUN_DIR.name}_{split}", exist_ok=True)
    print(f"\nОценка на выборке '{split}':")
    print(f"mAP@0.5      = {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 = {metrics.box.map:.4f}")
    print(f"Precision    = {metrics.box.mp:.4f}")
    print(f"Recall       = {metrics.box.mr:.4f}")
    names = model.names
    if len(names) > 1:
        print("\nmAP@0.5:0.95 по классам:")
        for i, c in enumerate(metrics.box.ap_class_index):
            print(f"  {names[int(c)]:<20} {metrics.box.maps[int(c)]:.4f}")


def train(epochs=50, batch=8, imgsz=640):
    has_test = prepare_data()
    model = YOLO(MODEL_NAME)
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=DEVICE,
        project=str(RUN_DIR.parent),
        name=RUN_DIR.name,
        exist_ok=True,
        patience=15,
        workers=2,
    )
    plot_results()
    evaluate(has_test)


def predict(sources, conf=0.25):
    model = YOLO(str(BEST_WEIGHTS))
    results = model.predict(source=sources, conf=conf, imgsz=640, device=DEVICE)
    out_dir = Path("predictions")
    out_dir.mkdir(exist_ok=True)
    for i, r in enumerate(results):
        print(f"\n{r.path}: найдено объектов {len(r.boxes)}")
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            print(f"  {model.names[int(b.cls)]:<15} {float(b.conf):.2f}  "
                  f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        img = r.plot()[:, :, ::-1]
        out_path = out_dir / f"result_{i + 1}.jpg"
        plt.imsave(out_path, img)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"YOLO12m: найдено объектов {len(r.boxes)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    print(f"\nРезультаты сохранены в папку {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="train", choices=["download", "train", "test", "plot", "predict"])
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--image", type=str, nargs="+", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.mode == "download":
        download(args.api_key)
    elif args.mode == "train":
        train(args.epochs, args.batch, args.imgsz)
    elif args.mode == "test":
        evaluate()
    elif args.mode == "plot":
        plot_results()
    else:
        if not args.image:
            parser.error("для predict укажите --image путь_или_ссылка")
        predict(args.image, args.conf)