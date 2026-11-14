import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(BASE_DIR, "weights", "best.pt")
TRACKERS_DIR = os.path.join(BASE_DIR, "trackers")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

CONFIDENCE = 0.1
IMAGE_SIZE = 640
SHORT_TRACK_FRAMES = 10
TRAIL_LENGTH = 30


def color_for_id(track_id):
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def draw_trails(frame, history):
    for track_id, points in history.items():
        if len(points) > 1:
            pts = np.array(points[-TRAIL_LENGTH:], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=False, color=color_for_id(track_id), thickness=3)


def run_tracking(video_path, tracker_cfg, save_path=None, show=False, window_title="Tracking"):
    if not os.path.exists(WEIGHTS):
        raise SystemExit(f"Веса детектора не найдены: {WEIGHTS}. Скопируйте best.pt из ЛР3 в папку weights.")
    model = YOLO(WEIGHTS)
    device = 0 if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    history = defaultdict(list)
    track_frames = defaultdict(int)
    track_classes = defaultdict(set)
    objects_per_frame = []
    frames = 0
    start = time.time()

    results = model.track(source=video_path, stream=True, persist=True, tracker=tracker_cfg, conf=CONFIDENCE,
                          imgsz=IMAGE_SIZE, device=device, verbose=False)
    for result in results:
        frames += 1
        boxes = result.boxes
        count = 0
        if boxes is not None and boxes.id is not None:
            for track_id, cls, xywh in zip(boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.xywh.tolist()):
                history[track_id].append((xywh[0], xywh[1]))
                track_frames[track_id] += 1
                track_classes[track_id].add(result.names[cls])
                count += 1
        objects_per_frame.append(count)

        if writer or show:
            frame = result.plot(line_width=2)
            draw_trails(frame, history)
            cv2.putText(frame, f"{os.path.basename(tracker_cfg)}  frame {frames}  objects {count}", (10, height - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            if writer:
                writer.write(frame)
            if show:
                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    elapsed = time.time() - start
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    lengths = list(track_frames.values())
    return {
        "frames": frames,
        "unique_ids": len(track_frames),
        "mean_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "short_tracks": sum(1 for n in lengths if n < SHORT_TRACK_FRAMES),
        "class_switch_tracks": sum(1 for c in track_classes.values() if len(c) > 1),
        "mean_objects_per_frame": float(np.mean(objects_per_frame)) if objects_per_frame else 0.0,
        "fps": frames / elapsed if elapsed else 0.0,
    }
