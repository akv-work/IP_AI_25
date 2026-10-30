from ultralytics import YOLO

model = YOLO("best.pt")

results = model.track(
    source="video.mp4",
    tracker="botsort_custom.yaml",
    conf=0.1,
    save=True,
    stream=True
)

for result in results:
    pass
