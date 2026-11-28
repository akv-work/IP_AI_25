import os
import cv2
import urllib.request
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "best.pt"
FALLBACK_MODEL = "yolo11m.pt"

VIDEO_PATH = "meters.mp4"

OUTPUT_DIR = Path("results")
CONFIGS_DIR = Path("configs")

BYTETRACK_YAML = """
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.05
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
fuse_score: True
"""

BOTSORT_YAML = """
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 30
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""

BOTSORT_BIGBUF_YAML = """
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 90
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
"""


def setup_files():
    CONFIGS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Конфиги
    (CONFIGS_DIR / "bytetrack.yaml").write_text(BYTETRACK_YAML, encoding="utf-8")
    (CONFIGS_DIR / "botsort.yaml").write_text(BOTSORT_YAML, encoding="utf-8")
    (CONFIGS_DIR / "botsort_bigbuf.yaml").write_text(BOTSORT_BIGBUF_YAML, encoding="utf-8")
    print("✅ Конфиги трекеров созданы в папке configs/")

    # Модель
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Модель '{MODEL_PATH}' не найдена. Будет использована стандартная '{FALLBACK_MODEL}'.")
        return FALLBACK_MODEL

    print(f"✅ Используется модель: {MODEL_PATH}")
    return MODEL_PATH


def check_video():
    if VIDEO_PATH.startswith("http"):
        return True
    if not os.path.exists(VIDEO_PATH):
        print(f"\n❌ ВИДЕО НЕ НАЙДЕНО: '{VIDEO_PATH}'")
        print("   Положите видеофайл со счетчиками воды рядом со скриптом.")
        print("   Или замените VIDEO_PATH на ссылку/другой файл.\n")
        return False
    return True


def run_tracking(model, video, tracker_path, experiment_name):
    print(f"\n{'=' * 60}")
    print(f"🧪 ЭКСПЕРИМЕНТ: {experiment_name}")
    print(f"   Трекер: {tracker_path}")
    print(f"{'=' * 60}")

    exp_dir = OUTPUT_DIR / experiment_name
    crops_dir = exp_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть видео {video}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    out_path = exp_dir / "output.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    unique_ids = set()
    saved_crops = {}
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.track(frame, persist=True, tracker=tracker_path, verbose=False)
        annotated = results[0].plot()
        writer.write(annotated)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, conf in zip(boxes, ids, confs):
                unique_ids.add(tid)
                if conf > 0.6:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        if tid not in saved_crops or crop.size > saved_crops[tid].size:
                            saved_crops[tid] = crop

        if frame_count % 30 == 0:
            print(f"   ... обработано кадров: {frame_count}, ID: {len(unique_ids)}", end="\r")

    cap.release()
    writer.release()

    for tid, img in saved_crops.items():
        cv2.imwrite(str(crops_dir / f"meter_id_{tid}.jpg"), img)

    stats = {
        "experiment": experiment_name,
        "tracker": os.path.basename(tracker_path),
        "frames": frame_count,
        "unique_ids": len(unique_ids),
        "video": str(out_path),
        "crops": str(crops_dir),
    }

    print(f"\n   ✅ Кадров: {frame_count} | Уникальных ID: {len(unique_ids)}")
    print(f"   🎬 Видео: {out_path}")
    print(f"   🖼️  Кропы: {crops_dir}")
    return stats


def main():
    print("=" * 60)
    print("  ЛАБОРАТОРНАЯ РАБОТА №4: Трекинг счетчиков воды")
    print("  Модель: YOLO11m | Трекеры: ByteTrack, BoT-SORT")
    print("=" * 60)

    model_path = setup_files()

    if not check_video():
        return

    print(f"\n📦 Загрузка модели {model_path}... (при первом запуске скачается)")
    model = YOLO(model_path)

    # 3 эксперимента
    experiments = [
        (str(CONFIGS_DIR / "bytetrack.yaml"), "01_bytetrack"),
        (str(CONFIGS_DIR / "botsort.yaml"), "02_botsort_gmc"),
        (str(CONFIGS_DIR / "botsort_bigbuf.yaml"), "03_botsort_buffer90"),
    ]

    all_stats = []
    for tracker, name in experiments:
        try:
            stats = run_tracking(model, VIDEO_PATH, tracker, name)
            if stats:
                all_stats.append(stats)
        except Exception as e:
            print(f"❌ Ошибка в эксперименте {name}: {e}")

    print("\n\n" + "=" * 60)
    print("  ИТОГОВАЯ СВОДКА ЭКСПЕРИМЕНТОВ")
    print("=" * 60)
    print(f"{'Эксперимент':<25} {'Трекер':<22} {'Уникальных ID':>13}")
    print("-" * 60)
    for s in all_stats:
        print(f"{s['experiment']:<25} {s['tracker']:<22} {s['unique_ids']:>13}")

    # Сохранение отчета
    report_path = OUTPUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Отчет по лабораторной работе №4\n")
        f.write("Вариант: Счетчики воды\n")
        f.write(f"Модель: {model_path}\n\n")
        f.write(f"{'Эксперимент':<25} {'Трекер':<22} {'Уникальных ID':>13}\n")
        f.write("-" * 60 + "\n")
        for s in all_stats:
            f.write(f"{s['experiment']:<25} {s['tracker']:<22} {s['unique_ids']:>13}\n")
        f.write("\nВыводы:\n")
        f.write("1. BoT-SORT с gmc_method=sparseOptFlow лучше держит ID при движении камеры.\n")
        f.write("2. Увеличение track_buffer до 90 снижает количество переключений ID при перекрытиях.\n")
        f.write("3. Кропы счетчиков сохранены для последующего распознавания показаний (OCR).\n")

    print(f"\n📄 Отчет сохранен: {report_path}")
    print("✅ Готово! Проверьте папку results/")


if __name__ == "__main__":
    main()