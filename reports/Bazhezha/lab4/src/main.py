import ssl
import urllib.request
from pathlib import Path

import certifi
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path.home() / "PycharmProjects" / "oivis_lab3" / "runs" / "train" / "weights" / "best.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SECONDS = 80
VIDEOS = {
    "yard": "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi",
    "crowd": "https://videos.pexels.com/video-files/853889/853889-sd_640_360_25fps.mp4",
}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(req, context=context) as response, dest.open("wb") as file:
        file.write(response.read())
    return dest


def clip(path: Path, seconds: int = SECONDS) -> Path:
    import cv2

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    keep = min(total, max(1, int(fps * seconds))) if total else int(fps * seconds)
    if total and keep >= total:
        cap.release()
        return path
    out = path.with_name(f"{path.stem}_{seconds}s.mp4")
    if out.exists() and out.stat().st_size > 0:
        cap.release()
        return out
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(cap.get(3)), int(cap.get(4))))
    for _ in range(keep):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    cap.release()
    writer.release()
    return out


def tracker_cfg(kind: str, **changes) -> Path:
    data = yaml.safe_load((ROOT / "trackers" / f"{kind}.yaml").read_text(encoding="utf-8"))
    data.update(changes)
    tag = kind if not changes else kind + "_" + "_".join(f"{key}-{value}" for key, value in changes.items())
    path = ROOT / "runs" / "cfgs" / f"{tag}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def track(model: YOLO, source: Path, cfg: Path, name: str) -> dict:
    lengths = {}
    counts = []
    confs = []
    shot = ROOT / "runs" / "frames" / f"{name}.jpg"
    shot.parent.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(
        model.track(
            source=str(source),
            tracker=str(cfg),
            stream=True,
            persist=False,
            save=True,
            project=str(ROOT / "runs"),
            name=name,
            exist_ok=True,
            device=DEVICE,
            imgsz=416,
            verbose=False,
        ),
        1,
    ):
        boxes = result.boxes
        ids = [] if boxes is None or boxes.id is None else boxes.id.int().tolist()
        for track_id in ids:
            lengths[track_id] = lengths.get(track_id, 0) + 1
        counts.append(len(ids))
        confs.append(float(boxes.conf.mean()) if boxes is not None and len(boxes) else 0.0)
        if index == 30:
            result.save(filename=str(shot))
    frames = len(counts)
    people = sum(counts) / frames if frames else 0
    tracks = len(lengths)
    life = sum(lengths.values()) / tracks if tracks else 0
    return {
        "name": name,
        "frames": frames,
        "ids": tracks,
        "people": people,
        "life": life,
        "frag": tracks / people if people else 0,
        "frame": shot,
        "counts": counts,
        "confs": confs,
    }


def plot(rows: list[dict]) -> list[Path]:
    import matplotlib.pyplot as plt

    folder = ROOT / "runs" / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    titles = {
        "yard_bytetrack": "yard ByteTrack",
        "yard_botsort": "yard BoT-SORT",
        "crowd_bytetrack": "crowd ByteTrack",
        "crowd_botsort": "crowd BoT-SORT",
        "crowd_bytetrack_track_buffer-5": "buffer 5",
        "crowd_bytetrack_track_buffer-90": "buffer 90",
        "crowd_bytetrack_track_high_thresh-0.55_new_track_thresh-0.55": "порог 0.55",
    }

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    for ax, video in zip(axes, ("yard", "crowd")):
        for kind in ("bytetrack", "botsort"):
            row = by_name(rows, f"{video}_{kind}")
            ax.plot(row["counts"], label=kind, linewidth=1.2)
        ax.set_title(video)
        ax.set_ylabel("Людей в кадре")
        ax.set_xlabel("Кадр")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    people = folder / "people.png"
    fig.savefig(people, dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    for ax, video in zip(axes, ("yard", "crowd")):
        for kind in ("bytetrack", "botsort"):
            row = by_name(rows, f"{video}_{kind}")
            ax.plot(row["confs"], label=kind, linewidth=1.2)
        ax.set_title(video)
        ax.set_ylabel("Средняя уверенность")
        ax.set_xlabel("Кадр")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    confidence = folder / "confidence.png"
    fig.savefig(confidence, dpi=120)
    plt.close(fig)

    labels = [titles[row["name"]] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, [row["ids"] for row in rows])
    axes[0].set_title("Уникальные id")
    axes[0].tick_params(axis="x", labelrotation=40)
    axes[1].bar(labels, [row["life"] for row in rows])
    axes[1].set_title("Средняя жизнь трека, кадры")
    axes[1].tick_params(axis="x", labelrotation=40)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    summary = folder / "summary.png"
    fig.savefig(summary, dpi=120)
    plt.close(fig)
    return [people, confidence, summary]


def by_name(rows: list[dict], name: str) -> dict:
    return next(row for row in rows if row["name"] == name)


def compare(base: dict, other: dict) -> str:
    def word(new, old, more, less, tol):
        if abs(new - old) <= tol:
            return "почти без изменений"
        return more if new > old else less

    return (
        f"{other['name']}: id {other['ids']} ({word(other['ids'], base['ids'], 'больше', 'меньше', 0)}, "
        f"у {base['name']} было {base['ids']}), жизнь {other['life']:.1f} "
        f"({word(other['life'], base['life'], 'длиннее', 'короче', 1)}, было {base['life']:.1f}), "
        f"людей/кадр {other['people']:.2f} (было {base['people']:.2f}), frag {other['frag']:.2f}."
    )


def report(rows: list[dict], videos: dict[str, str], plots: list[Path]) -> None:
    lines = [
        "Лабораторная работа № 4. Трекинг множественных объектов",
        "Детектор: YOLO11n из ЛР3, веса oivis_lab3/runs/train/weights/best.pt (People Detection).",
        "Трекеры Ultralytics: BoT-SORT (trackers/botsort.yaml) и ByteTrack (trackers/bytetrack.yaml).",
        "Метрики без эталонных треков: id — число разных идентификаторов, жизнь — средняя длина трека в кадрах,",
        "люди/кадр — среднее число сопровождаемых людей, frag = id / люди/кадр (ближе к 1 — меньше дробления).",
        "",
        "Видео из сети, класс person. Ролик длиннее 80 с обрезается, оба выбранных короче:",
    ]
    lines += [f"- {name}: {url}" for name, url in videos.items()]
    lines += ["", "Прогоны:"]
    for row in rows:
        lines.append(
            f"- {row['name']}: кадров {row['frames']}, id {row['ids']}, людей/кадр {row['people']:.2f}, "
            f"жизнь {row['life']:.1f}, frag {row['frag']:.2f}. "
            f"Кадр runs/frames/{row['frame'].name}, видео runs/{row['name']}/."
        )
    yard = by_name(rows, "yard_bytetrack")
    crowd = by_name(rows, "crowd_bytetrack")
    short = by_name(rows, "crowd_bytetrack_track_buffer-5")
    longer = by_name(rows, "crowd_bytetrack_track_buffer-90")
    strict = by_name(rows, "crowd_bytetrack_track_high_thresh-0.55_new_track_thresh-0.55")
    lines += [
        "",
        "Сравнение алгоритмов на одном детекторе:",
        compare(yard, by_name(rows, "yard_botsort")),
        compare(crowd, by_name(rows, "crowd_botsort")),
        "На неподвижной камере BoT-SORT и ByteTrack близки: оба сопоставляют рамки по IoU.",
        "Отличие BoT-SORT — компенсация движения камеры (gmc_method: sparseOptFlow). ReID выключен (with_reid: false).",
        "На yard frag большой, потому что за ролик через кадр проходит много разных людей, а одновременно их около 5.",
        "На crowd люди дольше остаются в кадре, поэтому frag ближе к 1 и лучше показывает дробление id.",
        "",
        "Влияние параметров ByteTrack на ролике crowd (база trackers/bytetrack.yaml: track_buffer 30, пороги 0.25):",
        compare(crowd, short),
        (
            "Буфер 5 кадров забывает человека после короткого перекрытия, поэтому один и тот же человек получает новый id."
            if short["ids"] > crowd["ids"]
            else "Буфер 5 кадров здесь почти не дробит треки: пропаданий длиннее 5 кадров мало."
        ),
        compare(crowd, longer),
        (
            "Буфер 90 кадров дольше хранит потерянный трек и реже выдаёт новый id."
            if longer["ids"] < crowd["ids"]
            else "Буфер 90 кадров совпал с буфером 30: в ролике нет пропаданий длиннее 30 кадров."
        ),
        compare(crowd, strict),
        "track_high_thresh — порог первого сопоставления, new_track_thresh — порог нового id. "
        "Детекция ниже new_track_thresh не открывает трек, но может продлить уже существующий, "
        "поэтому на кадре иногда остаётся рамка с низкой уверенностью.",
        "",
        "Графики работы сети:",
    ]
    notes = {
        "people.png": "сколько людей сеть ведёт в каждом кадре",
        "confidence.png": "средняя уверенность детекций по кадрам",
        "summary.png": "число id и средняя длина трека по всем прогонам",
    }
    lines += [f"{path.relative_to(ROOT)} — {notes[path.name]}." for path in plots]
    (ROOT / "отчет.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"Нет весов ЛР3: {WEIGHTS}")
    sources = {name: clip(download(url, ROOT / "samples" / Path(url).name)) for name, url in VIDEOS.items()}
    model = YOLO(WEIGHTS)
    print("классы:", model.names, "device:", DEVICE)
    rows = []
    jobs = [(video, kind, {}) for video in sources for kind in ("bytetrack", "botsort")]
    jobs += [
        ("crowd", "bytetrack", {"track_buffer": 5}),
        ("crowd", "bytetrack", {"track_buffer": 90}),
        ("crowd", "bytetrack", {"track_high_thresh": 0.55, "new_track_thresh": 0.55}),
    ]
    for video, kind, changes in jobs:
        cfg = tracker_cfg(kind, **changes)
        name = f"{video}_{cfg.stem}"
        print("трекинг", name)
        row = track(model, sources[video], cfg, name)
        rows.append(row)
        print(f"  id={row['ids']} люди/кадр={row['people']:.2f} жизнь={row['life']:.1f} frag={row['frag']:.2f}")
    plots = plot(rows)
    report(rows, VIDEOS, plots)
    print("графики:", ", ".join(str(path) for path in plots))
    print("отчет:", ROOT / "отчет.txt")


if __name__ == "__main__":
    main()
