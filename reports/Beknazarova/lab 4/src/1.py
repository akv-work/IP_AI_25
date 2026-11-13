import cv2
import os
from ultralytics import YOLO

MODEL_PATH = r"D:\\Учёба\\4 курс\\ОИвИС\\4\\best.pt"
VIDEO_PATH = r"D:\\Учёба\\4 курс\\ОИвИС\\4\\1.mp4"
OUTPUT_DIR = r"D:\\Учёба\\4 курс\\ОИвИС\\4\\output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_tracking(tracker_type="bytetrack"):
    print(f"Запуск трекинга с алгоритмом: {tracker_type}")
    
    model = YOLO(MODEL_PATH)
    
    tracker_file = "botsort_custom.yaml" if tracker_type == "botsort" else "bytetrack_custom.yaml"
    
    results = model.track(
        source=VIDEO_PATH,
        tracker=tracker_file,
        conf=0.35,
        iou=0.5,
        persist=True,
        verbose=False,
        stream=True
    )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    
    out_video_path = os.path.join(OUTPUT_DIR, f"track_{tracker_type}_final.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    for r in results:
        frame = r.orig_img.copy()
        
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None]*len(boxes)
            confs = r.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                area_ratio = (width * height) / (w * h) 

                if 0.00001 < area_ratio < 0.05:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID:{int(track_id) if track_id else 0} {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    out.release()
    print(f"Готово. Видео сохранено: {out_video_path}")

if __name__ == "__main__":
    run_tracking("bytetrack")
    run_tracking("botsort")