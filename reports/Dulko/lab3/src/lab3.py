import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    folder = Path(__file__).resolve().parent
    results = folder / "results"
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=folder / "data" / "data.yaml")
    parser.add_argument("--source", default=str(folder / "images"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "train":
        model = YOLO(str(args.weights or folder / "yolo11s.pt"))
        model.train(
            data=str(args.data.resolve()), epochs=args.epochs, imgsz=416,
            batch=16, workers=0, seed=42, project=str(results), name="train", exist_ok=True,
        )
        model = YOLO(str(model.trainer.best))
        model.val(
            data=str(args.data.resolve()), split="test", imgsz=416,
            batch=16, workers=0, project=str(results), name="test", exist_ok=True,
        )
    else:
        weights = args.weights or results / "train" / "weights" / "best.pt"
        YOLO(str(weights)).predict(
            source=args.source, conf=0.25, iou=0.5, imgsz=416, save=True,
            project=str(results), name="predict", exist_ok=True,
        )


if __name__ == "__main__":
    main()
