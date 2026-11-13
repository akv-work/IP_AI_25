import os
import cv2
from ultralytics import YOLO

def main():
    model = YOLO("best.pt")

    input_folder = "images"     
    output_folder = "results"  
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Ошибка: Папка '{input_folder}' не найдена!")
        return

    images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Найдено картинок для обработки: {len(images)}")

    for img_name in images:
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)
        
        print(f"Обработка...")
        
        results = model.predict(
            source=input_path,
            conf=0.25,      
            save=False      
        )
        
        annotated_frame = results[0].plot()
        cv2.imwrite(output_path, annotated_frame)

    print(f"\nВсе обработанные картинки сохранены в папку: {output_folder}")

if __name__ == "__main__":
    main()