import os
import yaml
import requests
from roboflow import Roboflow
from ultralytics import YOLO


ROBOFLOW_API_KEY = "bJcvJW4pC5nsQL8IOIS8"
WORKSPACE = "koer3741-gmail-com"
PROJECT_NAME = "watermeteramrv2"
VERSION = 1

EPOCHS = 15
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = "0"


def main():

    print("\n[1/4] Загрузка датасета с Roboflow...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
    dataset = project.version(VERSION).download("yolov8")

    dataset_path = dataset.location
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    print(f"Датасет скачан в: {dataset_path}")

    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    print(f"Классы в датасете: {data_config.get('names', [])}")


    print("\n[2/4] Начало обучения YOLO11m...")
    model = YOLO("yolo11m.pt")


    train_results = model.train(
        data=data_yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        project="runs_lab3",
        name="train_yolo11m",
        exist_ok=True,
        verbose=True
    )

    best_weights = os.path.join(train_results.save_dir, "weights", "best.pt")
    print(f"Обучение завершено. Лучшие веса: {best_weights}")


    print("\n[3/4] Оценка модели на тестовой выборке...")
    trained_model = YOLO(best_weights)

    # Валидация на тестовом сплите
    metrics = trained_model.val(
        data=data_yaml_path,
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE
    )

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ (mAP):")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print("=" * 40 + "\n")


    print("[4/4] Визуализация работы детектора...")

    # Скачиваем тестовую картинку счётчика воды из интернета
    test_img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8b/Water_meter_1.jpg/640px-Water_meter_1.jpg"
    local_img_path = "test_water_meter.jpg"

    try:
        print("Скачивание изображения...")
        response = requests.get(test_img_url, timeout=10)
        response.raise_for_status()
        with open(local_img_path, "wb") as f:
            f.write(response.content)
        print(f"Изображение сохранено: {local_img_path}")
    except Exception as e:
        print(f"Ошибка скачивания: {e}. Используем картинку из датасета.")

        import glob
        local_img_path = glob.glob(os.path.join(dataset_path, "test", "images", "*.jpg"))[0]


    predict_results = trained_model.predict(
        source=local_img_path,
        imgsz=IMG_SIZE,
        conf=0.25,
        save=True,
        project="runs_lab3",
        name="predict",
        exist_ok=True
    )

    print("\n" + "=" * 40)
    print("РЕЗУЛЬТАТЫ ДЕТЕКЦИИ:")
    for r in predict_results:
        print(f"Найдено объектов: {len(r.boxes)}")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f" - Класс: {trained_model.names[cls_id]}, уверенность: {conf:.2f}")

    print(f"\nКартинка с наложенными рамками сохранена в: runs_lab3/predict/")
    print("Готово! Все пункты лабораторной выполнены.")


if __name__ == "__main__":
    main()