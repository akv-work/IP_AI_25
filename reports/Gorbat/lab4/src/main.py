import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
LAB3 = ROOT.parent / "lab3"
WEIGHTS = LAB3 / "runs" / "train" / "weights" / "best.pt"
VIDEO = ROOT / "traffic.mp4"


def video_path():
    if VIDEO.exists():
        return VIDEO
    frames = {}
    for p in LAB3.joinpath("dataset").rglob("*.jpg"):
        if "adit_mp4-" not in p.name or "aditganteng" in p.name:
            continue
        m = re.search(r"adit_mp4-(\d+)", p.name)
        if m:
            frames[int(m.group(1))] = p
    keys = sorted(frames)
    keys = keys[:180]
    first = cv2.imread(str(frames[keys[0]]))
    h, w = first.shape[:2]
    wr = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for k in keys:
        wr.write(cv2.imread(str(frames[k])))
    wr.release()
    return VIDEO


def track(source):
    model = YOLO(str(WEIGHTS if WEIGHTS.exists() else "yolo11s.pt"))
    tracks = defaultdict(list)
    counts = []
    writer = None
    mid = None
    for i, r in enumerate(
        model.track(
            str(source), persist=True, stream=True, tracker="bytetrack.yaml",
            device="mps", conf=0.25,
        )
    ):
        n = 0
        if r.boxes is not None and r.boxes.id is not None:
            for tid, box, c in zip(
                r.boxes.id.int().tolist(),
                r.boxes.xyxy.cpu().tolist(),
                r.boxes.cls.int().tolist(),
            ):
                tracks[int(tid)].append((i, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, int(c)))
                n += 1
        counts.append(n)
        frame = r.plot()
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(ROOT / "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        writer.write(frame)
        if mid is None and n:
            mid = frame
    if writer:
        writer.release()
    if mid is not None:
        cv2.imwrite(str(ROOT / "preview.png"), mid)
    return tracks, counts


def plots(tracks, counts):
    plt.figure(figsize=(8, 5))
    for pts in tracks.values():
        plt.plot([p[1] for p in pts], [p[2] for p in pts], linewidth=1)
    plt.gca().invert_yaxis()
    plt.xlabel("x, px")
    plt.ylabel("y, px")
    plt.title("Траектории ТС (YOLO11s + ByteTrack)")
    plt.tight_layout()
    plt.savefig(ROOT / "trajectories.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(counts)
    plt.xlabel("кадр")
    plt.ylabel("число объектов")
    plt.title("Количество отслеживаемых объектов")
    plt.tight_layout()
    plt.savefig(ROOT / "counts.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist([len(v) for v in tracks.values()] or [0], bins=12)
    plt.xlabel("длина трека, кадры")
    plt.ylabel("число треков")
    plt.title("Длительность треков")
    plt.tight_layout()
    plt.savefig(ROOT / "track_lengths.png")
    plt.close()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else video_path()
    tracks, counts = track(src)
    plots(tracks, counts)
    (ROOT / "metrics.txt").write_text(
        f"tracks={len(tracks)}\nframes={len(counts)}\n"
        f"max_objects={max(counts) if counts else 0}\n"
        f"mean_objects={sum(counts) / len(counts) if counts else 0:.2f}\n"
    )
    print(f"tracks={len(tracks)} frames={len(counts)} max={max(counts) if counts else 0}")
