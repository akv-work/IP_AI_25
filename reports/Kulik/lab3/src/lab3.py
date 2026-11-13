import cv2
import supervision as sv
from roboflow import Roboflow
from ultralytics import YOLO


def main():
    rf = Roboflow(api_key="HeIANVEKRJVoonFxk9qG")
    project = rf.workspace("leo-ueno").project("people-detection-o4rdr")
    version = project.version(10)
    dataset = version.download("yolov8")

    model = YOLO("yolo12n.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=30,
        imgsz=512,
        batch=16,
        name="people_yolo12n",
        patience=10,
        workers=16,
        cache='disk'
    )

    model = YOLO("runs/detect/people_yolo12n/weights/best.pt")
    metrics = model.val(data=f"{dataset.location}/data.yaml", split="test")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

    model = YOLO("runs/detect/people_yolo12n/weights/best.pt")
    image = cv2.imread("3laba.jpg")
    
    if image is None:
        print("Ошибка: Не удалось найти картинку '3laba.jpg'.")
        return

    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = image.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    sv.plot_image(annotated)


if __name__ == "__main__":
    main()