import shutil
import sys
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
BEST = ROOT / "runs" / "train" / "weights" / "best.pt"
WEB = ROOT / "web.jpg"
WEB_URL = "https://ultralytics.com/images/bus.jpg"


def _download(patterns):
    snapshot_download(
        "LibreYOLO/vehicles-q0x2v",
        repo_type="dataset",
        local_dir=str(DATA),
        allow_patterns=patterns,
        max_workers=32,
    )


def _paired(split):
    img_dir, lbl_dir = DATA / split / "images", DATA / split / "labels"
    if not img_dir.exists():
        return []
    return [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and (lbl_dir / f"{p.stem}.txt").exists()
    ]


def data_yaml():
    if not (DATA / "data.yaml").exists() or not (DATA / "train" / "labels").exists():
        _download(["*.yaml", "*.txt", "README*", "train/labels/*", "valid/labels/*", "test/**"])
    if len(_paired("train")) < 200:
        _download(["train/images/*", "valid/**"])
    train_imgs = _paired("train")
    val_imgs = _paired("valid")
    if len(val_imgs) < 20 and len(train_imgs) > 40:
        val_dir_i, val_dir_l = DATA / "valid" / "images", DATA / "valid" / "labels"
        val_dir_i.mkdir(parents=True, exist_ok=True)
        val_dir_l.mkdir(parents=True, exist_ok=True)
        for p in train_imgs[::5]:
            shutil.copy2(p, val_dir_i / p.name)
            shutil.copy2(DATA / "train" / "labels" / f"{p.stem}.txt", val_dir_l / f"{p.stem}.txt")
    lines = []
    for line in (DATA / "data.yaml").read_text().splitlines():
        lines.append(f"path: {DATA.resolve()}" if line.startswith("path:") else line)
    path = DATA / "data.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def plot_loss(csv_path, out):
    import csv
    import matplotlib.pyplot as plt

    epochs, loss = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            epochs.append(float(row["epoch"]))
            loss.append(float(row["train/box_loss"]))
    plt.plot(epochs, loss)
    plt.xlabel("epoch")
    plt.ylabel("box_loss")
    plt.title("YOLO11s training loss")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def predict_image(source):
    YOLO(BEST).predict(source, save=True, project=str(ROOT / "runs"), name="predict", exist_ok=True)


def train():
    yaml = data_yaml()
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(yaml),
        epochs=5,
        imgsz=640,
        batch=8,
        device="mps",
        workers=0,
        project=str(ROOT / "runs"),
        name="train",
        exist_ok=True,
    )
    metrics = YOLO(BEST).val(data=str(yaml), split="test", device="mps", workers=0)
    (ROOT / "metrics.txt").write_text(
        f"mAP50={metrics.box.map50:.4f}\nmAP50-95={metrics.box.map:.4f}\n"
    )
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    plot_loss(ROOT / "runs" / "train" / "results.csv", ROOT / "loss.png")
    if not WEB.exists():
        try:
            urlretrieve(WEB_URL, WEB)
        except Exception as e:
            print("web image skip:", e)
    samples = list((DATA / "test" / "images").glob("*"))[:4]
    if WEB.exists():
        samples.append(WEB)
    YOLO(BEST).predict(samples, save=True, project=str(ROOT / "runs"), name="vis", exist_ok=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        train()
