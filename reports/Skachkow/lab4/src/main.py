from ultralytics import YOLO
import cv2
import os
model = YOLO("cats_yolo11n_best.pt")

VIDEO = "cats.mp4"
MAX_FRAMES = 1000

def track_limited(tracker_name, output_name):
    print(f"\n=== Трекер: {tracker_name} (первые {MAX_FRAMES} кадров) ===")

    cap = cv2.VideoCapture(VIDEO)
    frame_count = 0
    unique_ids = set()

    # Подготовка для сохранения видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(f"results/{output_name}.mp4", fourcc, fps, (width, height))

    while cap.isOpened() and frame_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            tracker=tracker_name,
            persist=True,
            conf=0.3,
            verbose=False
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            unique_ids.update(results[0].boxes.id.int().tolist())

        annotated = results[0].plot()
        out.write(annotated)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Обработано кадров: {frame_count}")

    cap.release()
    out.release()
    print(f"Уникальных ID: {len(unique_ids)}")
    print(f"Видео сохранено: results/{output_name}.mp4")



os.makedirs("results", exist_ok=True)

track_limited("bytetrack.yaml", "bytetrack")
track_limited("botsort.yaml", "botsort")

print("\nГотово.")