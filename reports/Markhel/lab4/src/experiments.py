"""
Исследование влияния параметров конфигурационных файлов BoT-SORT и ByteTrack на качество трекинга.
Для каждого видео из папки videos запускается трекинг со всеми конфигурациями из EXPERIMENTS.
"""
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from tracking import RESULTS_DIR, TRACKERS_DIR, VIDEOS_DIR, run_tracking

EXPERIMENTS_DIR = os.path.join(TRACKERS_DIR, "experiments")

EXPERIMENTS = {
    "bytetrack": ("bytetrack", {}),
    "bytetrack_high0.5": ("bytetrack", {"track_high_thresh": 0.5}),
    "bytetrack_new0.6": ("bytetrack", {"new_track_thresh": 0.6}),
    "bytetrack_buffer5": ("bytetrack", {"track_buffer": 5}),
    "bytetrack_buffer90": ("bytetrack", {"track_buffer": 90}),
    "bytetrack_match0.5": ("bytetrack", {"match_thresh": 0.5}),
    "bytetrack_match0.95": ("bytetrack", {"match_thresh": 0.95}),
    "botsort": ("botsort", {}),
    "botsort_high0.5": ("botsort", {"track_high_thresh": 0.5}),
    "botsort_buffer90": ("botsort", {"track_buffer": 90}),
    "botsort_gmc_none": ("botsort", {"gmc_method": "none"}),
    "botsort_reid": ("botsort", {"with_reid": True}),
}
SAVE_VIDEO_FOR = {"bytetrack", "botsort"}
METRICS = [
    ("unique_ids", "Уникальных ID"),
    ("short_tracks", "Коротких треков (< 10 кадров)"),
    ("mean_track_length", "Средняя длина трека, кадров"),
    ("fps", "Скорость, кадров/с"),
]


def write_config(name, base, overrides):
    with open(os.path.join(TRACKERS_DIR, f"{base}.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update(overrides)
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    path = os.path.join(EXPERIMENTS_DIR, f"{name}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def plot_results(rows, videos, path):
    names = list(EXPERIMENTS)
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(14, 4 * len(METRICS)))
    width = 0.8 / len(videos)
    x = np.arange(len(names))
    for ax, (key, title) in zip(axes, METRICS):
        for i, video in enumerate(videos):
            values = [next(r[key] for r in rows if r["video"] == video and r["config"] == n) for n in names]
            ax.bar(x + (i - (len(videos) - 1) / 2) * width, values, width, label=video)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Влияние параметров трекеров на качество трекинга")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def main():
    videos = sorted(f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")))
    if not videos:
        raise SystemExit(f"В папке {VIDEOS_DIR} нет видео.")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for name, (base, overrides) in EXPERIMENTS.items():
        cfg_path = write_config(name, base, overrides)
        for video in videos:
            stem = os.path.splitext(video)[0]
            save_path = os.path.join(RESULTS_DIR, "videos", f"{stem}_{name}.mp4") if name in SAVE_VIDEO_FOR else None
            stats = run_tracking(os.path.join(VIDEOS_DIR, video), cfg_path, save_path=save_path)
            rows.append({"config": name, "video": video, **stats})
            print(f"{name:<20} {video:<35} ID={stats['unique_ids']:<4} short={stats['short_tracks']:<4} "
                  f"len={stats['mean_track_length']:<7.1f} switch={stats['class_switch_tracks']:<3} "
                  f"obj/frame={stats['mean_objects_per_frame']:.2f} fps={stats['fps']:.1f}", flush=True)

    with open(os.path.join(RESULTS_DIR, "experiments.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "experiments.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    plot_results(rows, videos, os.path.join(RESULTS_DIR, "experiments.png"))
    plt.show()


if __name__ == "__main__":
    main()
