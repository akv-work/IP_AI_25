import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

DATASET_YAML = r"D:\\Учёба\\4 курс\\ОИвИС\\3\\dataset\\data.yaml"

BEST_WEIGHTS = r"D:\\Учёба\\4 курс\\ОИвИС\\3\\runs\\detect\\runs\\license_plate_yolo11n\\weights\\best.pt"

PROJECT_DIR = r"D:\\Учёба\\4 курс\\ОИвИС\\3\\runs\\gui_results"

EPOCHS = 50
BATCH_SIZE = 4
IMG_SIZE = 512
DEVICE = 0

def train_model():
    print("Запуск обучения.")
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name="license_plate_yolo11n_new", 
        workers=2,
        patience=15,
        plots=True,
        val=False,
        amp=False
    )
    print("Обучение завершено.")
    return os.path.join(PROJECT_DIR, "train", "license_plate_yolo11n_new", "weights", "best.pt")

def evaluate_model():
    print("Запуск валидации на тестовой выборке.")
    trained_model = YOLO(BEST_WEIGHTS)
    metrics = trained_model.val(
        data=DATASET_YAML,
        split="test",
        device=DEVICE,
        project=PROJECT_DIR,
        name="test_evaluation"
    )

    print("\nРезультаты валидации:")
    print("mAP50:", round(metrics.box.map50, 4))
    print("mAP50-95:", round(metrics.box.map, 4))
    print("Precision:", round(metrics.box.mp, 4))
    print("Recall:", round(metrics.box.mr, 4))

def launch_gui():
    print("Запуск графического интерфейса.")
    root = tk.Tk()
    root.title("Детектор номерных знаков")
    root.geometry("300x150")
    root.eval("tk::PlaceWindow . center")

    def select_and_predict():
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            print("Выбрано изображение:", file_path)
            model = YOLO(BEST_WEIGHTS)
            model.predict(
                source=file_path,
                conf=0.5,
                save=True,
                project=PROJECT_DIR,
                name="gui_predictions"
            )
            print("Результат сохранен в папку:", os.path.join(PROJECT_DIR, "gui_predictions"))

    btn = tk.Button(root, text="Выбрать фото и найти номер", command=select_and_predict, height=2, width=30)
    btn.pack(pady=50)

    root.mainloop()

if __name__ == "__main__":
    if os.path.exists(BEST_WEIGHTS):
        print(f"Найдена обученная модель: {BEST_WEIGHTS}")
        choice = input("Переобучить модель заново? (y/n): ").strip().lower()
        if choice == 'y':
            BEST_WEIGHTS = train_model()
        else:
            print("Пропуск обучения, используются готовые веса.")
    else:
        print(f"Модель не найдена по пути: {BEST_WEIGHTS}")
        print("Запуск обучения с нуля...")
        BEST_WEIGHTS = train_model()

    if os.path.exists(BEST_WEIGHTS):
        evaluate_model()
        launch_gui()
    else:
        print("Ошибка: файл весов не найден.")