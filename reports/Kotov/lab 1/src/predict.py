import tkinter as tk
from tkinter import filedialog

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import SimpleCNN, CLASSES

MODEL_PATH = 'stl10_cnn.pth'
device = torch.device('cpu')


def choose_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title='Выбери изображение для классификации',
        filetypes=[('Изображения', '*.jpg *.jpeg *.png *.bmp *.webp'), ('Все файлы', '*.*')]
    )
    root.destroy()
    return file_path


def main():
    file_path = choose_image()
    if not file_path:
        print('Файл не выбран.')
        return

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    orig_image = Image.open(file_path).convert('RGB')
    input_tensor = transform(orig_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, 3)

    best_class = CLASSES[top_idxs[0].item()]
    best_prob = top_probs[0].item() * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    ax1.imshow(orig_image)
    ax1.set_title('Исходное изображение')
    ax1.axis('off')

    ax2.imshow(orig_image.resize((96, 96)))
    ax2.set_title(f'Прогноз: {best_class} ({best_prob:.1f}%)', fontsize=12, fontweight='bold')
    ax2.axis('off')

    top3_text = 'Топ-3 класса:\n' + '\n'.join(
        f'{CLASSES[idx.item()]}: {prob.item() * 100:.1f}%'
        for prob, idx in zip(top_probs, top_idxs)
    )
    fig.text(0.52, 0.04, top3_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.show()


if __name__ == '__main__':
    main()