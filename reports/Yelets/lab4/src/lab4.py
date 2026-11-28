import cv2
import os
from ultralytics import YOLO

VIDEO_PATH = "17097260_2160_3840_60fps.mp4" 
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("best.pt")

def run_tracking(tracker_type="bytetrack"):
    print(f"\nЗапуск трекинга: {tracker_type}...")
    
    tracker_file = "botsort_custom.yaml" if tracker_type == "botsort" else "bytetrack_custom.yaml"
    
    results = model.track(
        source=VIDEO_PATH,
        tracker=tracker_file,
        conf=0.35,        
        iou=0.6,         
        imgsz=1280,    
        persist=True,
        verbose=False,
        stream=True
    )
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    
    w_out, h_out = w // 2, h // 2 
    
    out_video_path = os.path.join(OUTPUT_DIR, f"FINAL_{tracker_type}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w_out, h_out))

    unique_ids = set()

    for r in results:
        frame = r.orig_img.copy()
        
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None]*len(boxes)
            confs = r.boxes.conf.cpu().numpy()

            for box, track_id, conf in zip(boxes, ids, confs):
                x1, y1, x2, y2 = map(int, box)
                
                color = (0, 255, 0) if tracker_type == "bytetrack" else (255, 0, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{int(track_id) if track_id else 0}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if track_id is not None:
                    unique_ids.add(int(track_id))

        frame_resized = cv2.resize(frame, (w_out, h_out))
        out.write(frame_resized)

    out.release()
    print(f"Уникальных ID: {len(unique_ids)}. Сохранено: {out_video_path}")

run_tracking("bytetrack")
run_tracking("botsort")