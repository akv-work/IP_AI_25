import os
from ultralytics import YOLO

model = YOLO("best.pt")

input_folder = "kartinki"     
output_folder = "results"  

os.makedirs(output_folder, exist_ok=True)

images = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"Найдено картинок для обработки: {len(images)}")

for img_name in images:
    input_path = os.path.join(input_folder, img_name)
    output_path = os.path.join(output_folder, img_name)
    
    print(f"Обработка: {img_name}...")
    
    results = model.predict(
        source=input_path,
        conf=0.2,     
        save=False      
    )
    
    annotated_frame = results[0].plot()
    
    import cv2
    cv2.imwrite(output_path, annotated_frame)

print(f"\nВсе обработанные картинки сохранены в папку: {output_folder}")