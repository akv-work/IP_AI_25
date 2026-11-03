import argparse
import json
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


def main():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--buffer", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--weights", type=Path, default=(
        folder.parents[1] / "lab3" / "src" / "results" / "train" / "weights" / "best.pt"
    ))
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    capture.release()

    for tracker in ["botsort", "bytetrack"]:
        output = folder.parent / "results" / Path(args.source).stem / f"{tracker}_{args.buffer}_{args.threshold}"
        output.mkdir(parents=True, exist_ok=True)
        config = yaml.safe_load((folder / f"{tracker}.yaml").read_text())
        config["track_buffer"] = args.buffer
        config["track_high_thresh"] = args.threshold
        config_path = output / f"{tracker}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        model = YOLO(str(args.weights))
        writer = None
        identities = set()
        frames = visible = observations = 0
        try:
            for result in model.track(
                source=args.source, tracker=str(config_path), stream=True,
                conf=0.1, iou=0.5, imgsz=640, verbose=False,
            ):
                frame = result.plot()
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output / "tracking.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Cannot create output video")
                writer.write(frame)
                frames += 1
                if result.boxes.id is not None:
                    identities.update(result.boxes.id.int().tolist())
                    visible += 1
                    observations += len(result.boxes.id)
        finally:
            if writer is not None:
                writer.release()
        scores = {
            "frames": frames, "frames_with_tracks": visible,
            "tracked_boxes": observations, "track_ids": len(identities),
        }
        (output / "metrics.json").write_text(json.dumps(scores, indent=2))
        print(tracker, args.buffer, scores)


if __name__ == "__main__":
    main()
