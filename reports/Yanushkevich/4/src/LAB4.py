import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
import ultralytics
import yaml
from ultralytics import YOLO

WEIGHTS = "runs/yolo12m_watermeter/weights/best.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"
CFG_DIR = Path("trackers_cfg")
OUT_DIR = Path("track_results")
DEFAULT_CFG_DIR = Path(ultralytics.__file__).parent / "cfg" / "trackers"


def make_cfg(base, name, **overrides):
    """Копия стандартного конфига (bytetrack / botsort) с изменёнными параметрами."""
    with open(DEFAULT_CFG_DIR / f"{base}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in overrides:
        if k not in cfg:
            print(f"[!] параметра '{k}' нет в {base}.yaml (версия ultralytics другая?) - всё равно записываю")
    cfg.update(overrides)
    CFG_DIR.mkdir(exist_ok=True)
    path = CFG_DIR / f"{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def run_tracking(source, cfg_path, tag, conf=0.25, iou=0.7, imgsz=640,
                 save_video=True, max_frames=None, weights=WEIGHTS):
    OUT_DIR.mkdir(exist_ok=True)
    model = YOLO(weights)

    cap = cv2.VideoCapture(str(source))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(OUT_DIR / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    track_len = defaultdict(int)
    history = defaultdict(list)
    per_frame_objs = []
    frames = 0
    t0 = time.time()

    results = model.track(source=str(source), tracker=str(cfg_path), conf=conf, iou=iou,
                          imgsz=imgsz, device=DEVICE, stream=True, persist=True, verbose=False)
    for r in results:
        frames += 1
        frame = r.plot()
        n_tracked = 0
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xywh = r.boxes.xywh.cpu().tolist()
            n_tracked = len(ids)
            for tid, (x, y, _, _) in zip(ids, xywh):
                track_len[tid] += 1
                history[tid].append((int(x), int(y)))
                pts = history[tid][-40:]
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(frame, a, b, (0, 255, 255), 2)
        per_frame_objs.append(n_tracked)
        if writer is not None:
            writer.write(frame)
        if max_frames and frames >= max_frames:
            break
    if writer is not None:
        writer.release()

    elapsed = time.time() - t0
    lens = list(track_len.values())
    max_concurrent = max(per_frame_objs) if per_frame_objs else 0
    stats = {
        "tag": tag,
        "frames": frames,
        "unique_ids": len(lens),
        "max_concurrent": max_concurrent,
        "id_overhead": max(len(lens) - max_concurrent, 0),
        "avg_track_len": round(statistics.mean(lens), 1) if lens else 0,
        "median_track_len": statistics.median(lens) if lens else 0,
        "short_tracks_lt10": sum(1 for x in lens if x < 10),
        "avg_objs_per_frame": round(statistics.mean(per_frame_objs), 2) if per_frame_objs else 0,
        "fps": round(frames / elapsed, 1) if elapsed else 0,
    }
    print(stats)
    return stats


def experiment_grid():
    """(базовый алгоритм, имя прогона, переопределяемые параметры)."""
    return [
        ("bytetrack", "byte_default", {}),
        ("bytetrack", "byte_buffer10", {"track_buffer": 10}),
        ("bytetrack", "byte_buffer90", {"track_buffer": 90}),
        ("bytetrack", "byte_match0.6", {"match_thresh": 0.6}),
        ("bytetrack", "byte_match0.95", {"match_thresh": 0.95}),
        ("bytetrack", "byte_high0.5", {"track_high_thresh": 0.5, "new_track_thresh": 0.5}),
        ("bytetrack", "byte_low0.01", {"track_low_thresh": 0.01}),
        ("botsort", "bot_default", {}),
        ("botsort", "bot_gmc_none", {"gmc_method": "none"}),
        ("botsort", "bot_buffer90", {"track_buffer": 90}),
        ("botsort", "bot_match0.6", {"match_thresh": 0.6}),
        ("botsort", "bot_prox0.2", {"proximity_thresh": 0.2}),
        ("botsort", "bot_high0.5", {"track_high_thresh": 0.5, "new_track_thresh": 0.5}),
        ("botsort", "bot_reid", {"with_reid": True}),
    ]


def experiment(source, conf, iou, imgsz, max_frames, save_video, only=None):
    rows = []
    for base, name, over in experiment_grid():
        if only and not any(o in name for o in only):
            continue
        print(f"\n=== {name}: {over or 'параметры по умолчанию'}")
        cfg = make_cfg(base, name, **over)
        try:
            st = run_tracking(source, cfg, name, conf, iou, imgsz, save_video, max_frames)
        except Exception as e:
            print(f"[!] прогон {name} не удался: {e}")
            continue
        st["algorithm"] = base
        st["params"] = str(over) if over else "default"
        rows.append(st)

    if not rows:
        return
    OUT_DIR.mkdir(exist_ok=True)
    csv_path = OUT_DIR / "experiment_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nТаблица: {csv_path}")
    plot_experiment(rows)


def plot_experiment(rows):
    tags = [r["tag"] for r in rows]
    colors = ["tab:blue" if r["algorithm"] == "bytetrack" else "tab:orange" for r in rows]
    metrics = [("unique_ids", "Число уникальных ID"),
               ("id_overhead", "Лишние ID (фрагментация)"),
               ("avg_track_len", "Средняя длина трека, кадров"),
               ("fps", "Скорость, кадров/с")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (key, title) in zip(axes.ravel(), metrics):
        ax.bar(tags, [r[key] for r in rows], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=70)
        ax.grid(axis="y")
    fig.suptitle("Синий - ByteTrack, оранжевый - BoT-SORT")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "experiment_plot.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["track", "experiment"])
    p.add_argument("--source", required=True, help="путь к видео")
    p.add_argument("--tracker", default="bytetrack", choices=["bytetrack", "botsort"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--only", nargs="*", default=None, help="фильтр прогонов по подстроке имени, напр. byte bot_reid")
    a = p.parse_args()

    if a.mode == "track":
        cfg = make_cfg(a.tracker, f"{a.tracker}_default")
        run_tracking(a.source, cfg, f"{a.tracker}_{Path(a.source).stem}", a.conf, a.iou, a.imgsz,
                     not a.no_video, a.max_frames)
    else:
        experiment(a.source, a.conf, a.iou, a.imgsz, a.max_frames, not a.no_video, a.only)
