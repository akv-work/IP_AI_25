"""
Лабораторная работа №4. Трекинг множественных объектов.
Вариант 6: детектор YOLO11m из ЛР3 (жесты «камень», «ножницы», «бумага»), трекеры BoT-SORT и ByteTrack.

Запуск:
    python track.py                              - выбор видео в окне, трекер BoT-SORT
    python track.py видео.mp4 bytetrack          - указанное видео и трекер (botsort | bytetrack | путь к .yaml)
Во время показа видео нажмите q, чтобы остановить.
"""
import os
import sys

from tracking import RESULTS_DIR, TRACKERS_DIR, VIDEOS_DIR, run_tracking


def choose_video():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Выберите видео",
        initialdir=VIDEOS_DIR if os.path.isdir(VIDEOS_DIR) else None,
        filetypes=[("Видео", "*.mp4 *.avi *.mov *.mkv"), ("Все файлы", "*.*")],
    )
    root.destroy()
    return path or None


def resolve_tracker(name):
    if name.endswith(".yaml"):
        return name
    return os.path.join(TRACKERS_DIR, f"{name}.yaml")


def main():
    video = sys.argv[1] if len(sys.argv) > 1 else choose_video()
    if not video:
        raise SystemExit("Видео не выбрано.")
    tracker = resolve_tracker(sys.argv[2] if len(sys.argv) > 2 else "botsort")

    name = f"{os.path.splitext(os.path.basename(video))[0]}_{os.path.splitext(os.path.basename(tracker))[0]}.mp4"
    save_path = os.path.join(RESULTS_DIR, "videos", name)
    stats = run_tracking(video, tracker, save_path=save_path, show=True,
                         window_title=f"{os.path.basename(video)} - {os.path.basename(tracker)}")

    print(f"Видео: {video}")
    print(f"Трекер: {tracker}")
    print(f"Кадров обработано: {stats['frames']}")
    print(f"Уникальных ID треков: {stats['unique_ids']}")
    print(f"Средняя длина трека: {stats['mean_track_length']:.1f} кадров")
    print(f"Коротких треков (< 10 кадров): {stats['short_tracks']}")
    print(f"Треков со сменой класса: {stats['class_switch_tracks']}")
    print(f"Среднее число объектов в кадре: {stats['mean_objects_per_frame']:.2f}")
    print(f"Скорость обработки: {stats['fps']:.1f} кадров/с")
    print(f"Результат сохранён: {save_path}")


if __name__ == "__main__":
    main()
