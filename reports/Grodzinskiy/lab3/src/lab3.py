import sys
from pathlib import Path
from urllib.request import urlretrieve
import csv
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
BEST = ROOT / "runs" / "train" / "weights" / "best.pt"
WEB = ROOT / "web.jpg"
WEB_URL = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=640"

def plot_loss(csv_path, out):
    epochs, loss = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v.strip() for k, v in row.items()}
            epochs.append(float(row["epoch"]))
            loss.append(float(row["train/box_loss"]))
    plt.plot(epochs, loss)
    plt.xlabel("epoch")
    plt.ylabel("box_loss")
    plt.title("YOLO11s training loss (Cats)")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def predict_image(source):
    YOLO(BEST).predict(source, save=True, project=str(ROOT / "runs"), name="predict", exist_ok=True)

def train():
    yaml = DATA / "data.yaml"
    if not yaml.exists():
        print(f"ОШИБКА: Файл {yaml} не найден!")
        return

    model = YOLO("yolo11s.pt")
    
    model.train(
        data=str(yaml),
        epochs=5,
        imgsz=640,
        batch=8,
        device="cpu", 
        workers=0,  
        project=str(ROOT / "runs"),
        name="train",
        exist_ok=True,
    )
    
    metrics = YOLO(BEST).val(data=str(yaml), split="test", device="cpu", workers=0)
    
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
    print("Готово! Результаты сохранены в папке runs/vis")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        train()