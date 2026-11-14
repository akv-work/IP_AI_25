import cv2
import os
from ultralytics import YOLO

MODEL_PATH = "best.pt"
VIDEO_PATH = "input_video.mp4"
OUTPUT_DIR = "runs" 

custom_tracker_config = """
tracker_type: bytetrack
track_high_thresh: 0.3
track_low_thresh: 0.1
new_track_thresh: 0.5
track_buffer: 60
match_thresh: 0.8
fuse_score: True
"""

with open("my_bytetrack.yaml", "w") as f:
    f.write(custom_tracker_config)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель {MODEL_PATH} не найдена.")

model = YOLO(MODEL_PATH)

def run_tracking(source, tracker_cfg, output_name):
    results = model.track(
        source=source,
        tracker=tracker_cfg,
        persist=True,
        conf=0.3,
        iou=0.5,
        show=False,
        save=True,
        project=OUTPUT_DIR,
        name=output_name,
        verbose=False
    )

if __name__ == "__main__":
    if not os.path.exists(VIDEO_PATH):
        print(f"Видео {VIDEO_PATH} не найдено.")
    else:
        run_tracking(
            source=VIDEO_PATH,
            tracker="botsort.yaml", 
            output_name="exp1_botsort"
        )

        run_tracking(
            source=VIDEO_PATH,
            tracker="my_bytetrack.yaml", 
            output_name="exp2_bytetrack_custom"
        )

        model.track(
            source=VIDEO_PATH,
            tracker="botsort.yaml",
            persist=True,
            conf=0.3,
            iou=0.6,
            save=True,
            project=OUTPUT_DIR,
            name="exp3_iou_0.6"
        )