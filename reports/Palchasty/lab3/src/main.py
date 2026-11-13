import random
from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import cv2
from ultralytics import YOLO
from roboflow import Roboflow

random.seed(42)


class PlateDetectorPipeline:
    def __init__(self, api_key="QUYMcDyS147x6eWX6IUw"):
        self.api_key = api_key
        self.rf = Roboflow(api_key=self.api_key)
        self.proj = self.rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
        self.ds = self.proj.version(11).download("yolov8")
        self.root = Path(self.ds.location)
        self.yaml_path = self.root / "data.yaml"
        self._prepare_yaml()
        self._report_counts()
        self.model = None
        self.save_dir = None
        self.best_model = None

    def _prepare_yaml(self):
        with open(self.yaml_path) as f:
            cfg = yaml.safe_load(f)
        print(f"Classes ({len(cfg['names'])}): {cfg['names']}")
        cfg["path"] = str(self.root)
        cfg["train"] = "train/images"

        valid_imgs = self.root / "valid" / "images"
        if valid_imgs.exists() and any(valid_imgs.glob("*")):
            cfg["val"] = "valid/images"
        else:
            print("valid/ пустая — использую test/images как val")
            cfg["val"] = "test/images"

        cfg["test"] = "test/images"

        with open(self.yaml_path, "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)
        self.cfg = cfg

    def _report_counts(self):
        for part in ("train", "valid", "test"):
            p = self.root / part / "images"
            cnt = sum(1 for _ in p.glob("*")) if p.exists() else 0
            print(f"{part}: {cnt}")

    def preview_batch(self, part="train", k=6):
        files = random.sample(list((self.root / part / "images").glob("*")), k)
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        for ax, fp in zip(axs.flat, files):
            im = Image.open(fp).convert("RGB")
            w, h = im.size
            ax.imshow(im)
            ax.axis("off")
            lbl = self.root / part / "labels" / (fp.stem + ".txt")
            if lbl.exists():
                for ln in lbl.read_text().strip().splitlines():
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    _, xc, yc, bw, bh = map(float, parts[:5])
                    ax.add_patch(patches.Rectangle(
                        ((xc - bw / 2) * w, (yc - bh / 2) * h),
                        bw * w, bh * h, fill=False, edgecolor="lime", linewidth=2))
        plt.suptitle("Sample training images with labels")
        plt.tight_layout()
        plt.show()

    def train(self, epochs=20):
        self.model = YOLO("yolo12n.pt")
        self.model.train(
            data=str(self.yaml_path),
            epochs=epochs,
            imgsz=640,
            batch=16,
            project="plate_training",
            name="run1",
            exist_ok=True,
            seed=42,
        )
        self.save_dir = Path(self.model.trainer.save_dir)
        print(f"Training output: {self.save_dir}")

    def plot_history(self):
        log = pd.read_csv(self.save_dir / "results.csv")
        log.columns = log.columns.str.strip()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(log["epoch"], log["train/box_loss"], label="train", marker="s", linestyle="--", markersize=3)
        ax1.plot(log["epoch"], log["val/box_loss"], label="val", marker="s", linestyle="--", markersize=3)
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Box loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(log["epoch"], log["metrics/mAP50(B)"], label="mAP@0.5", marker="^", linestyle="-.", markersize=3)
        ax2.plot(log["epoch"], log["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", marker="^", linestyle="-.",
                 markersize=3)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("mAP")
        ax2.set_title("Validation mAP")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def evaluate(self):
        self.best_model = YOLO(str(self.save_dir / "weights" / "best.pt"))
        m = self.best_model.val(data=str(self.yaml_path), split="test", imgsz=640)
        print(f"P={m.box.mp:.4f}, R={m.box.mr:.4f}, mAP50={m.box.map50:.4f}, mAP50-95={m.box.map:.4f}")
        return m

    def process_videos(self, sources, dst="output_combined.mp4", conf=0.2, show=False):
        caps = [cv2.VideoCapture(str(s)) for s in sources]
        for i, cap in enumerate(caps):
            if not cap.isOpened():
                raise FileNotFoundError(f"Не открывается видео: {sources[i]}")

        fps = caps[0].get(cv2.CAP_PROP_FPS) or 25
        w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        total_frames = 0
        total_dets = 0

        for cap, src_name in zip(caps, sources):
            print(f"--- Обработка: {src_name} ---")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))

                res = self.best_model.predict(frame, conf=conf, imgsz=640, max_det=100, verbose=False)[0]

                for box in res.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    score = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"plate {score:.2f}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 0), 2)
                    total_dets += 1

                writer.write(frame)
                total_frames += 1

                if show:
                    cv2.imshow("detect", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            cap.release()

        writer.release()
        if show:
            cv2.destroyAllWindows()

        print(f"Готово! Кадров обработано: {total_frames}, детекций: {total_dets}")
        print(f"Единое видео сохранено: {Path(dst).resolve()}")

    def run_all(self):
        self.preview_batch("train")
        self.train(20)
        self.plot_history()
        self.evaluate()

        video_paths = [
            r"C:\Users\arosl\PycharmProjects\OIS3\1.mp4",
            r"C:\Users\arosl\PycharmProjects\OIS3\2.mp4"
        ]

        self.process_videos(video_paths, "output_combined.mp4")


if __name__ == "__main__":
    pipe = PlateDetectorPipeline()
    pipe.run_all()