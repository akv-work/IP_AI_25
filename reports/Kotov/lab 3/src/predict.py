import glob
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from ultralytics import YOLO


def find_best_weights():
    candidates = glob.glob(os.path.join('runs', 'detect', '*', 'weights', 'best.pt'))
    if not candidates:
        raise FileNotFoundError('Веса best.pt не найдены. Сначала запусти train.py.')
    return max(candidates, key=os.path.getmtime)


def choose_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title='Выбери фото кота для детекции',
        filetypes=[('Изображения', '*.jpg *.jpeg *.png *.bmp *.webp'), ('Все файлы', '*.*')]
    )
    root.destroy()
    return file_path


def main():
    weights_path = find_best_weights()
    file_path = choose_image()
    if not file_path:
        print('Файл не выбран.')
        return

    model = YOLO(weights_path)
    results = model.predict(source=file_path, conf=0.25, verbose=False)
    plotted_rgb = results[0].plot()[:, :, ::-1]

    plt.figure(figsize=(8, 6))
    plt.imshow(plotted_rgb)
    plt.title(f'Детекция YOLO12s (найдено объектов: {len(results[0].boxes)})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()