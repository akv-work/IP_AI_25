from ultralytics import YOLO

def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        patience=20,
        plots=True,
        project="runs/detect",
        name="people_yolo26n"
    )

if __name__ == "__main__":
    main()
