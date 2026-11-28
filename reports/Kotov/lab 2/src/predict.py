import tkinter as tk
from tkinter import filedialog

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import SimpleCNN, CLASSES
from squeezenet_model import build_squeezenet

CUSTOM_MODEL_PATH = 'stl10_cnn.pth'
PRETRAINED_MODEL_PATH = 'stl10_squeezenet.pth'
device = torch.device('cpu')


def choose_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(
        title='Выбери изображение для сравнения моделей',
        filetypes=[('Изображения', '*.jpg *.jpeg *.png *.bmp *.webp'), ('Все файлы', '*.*')]
    )
    root.destroy()
    return file_path


def get_top3(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, 3)
    labels = [CLASSES[idx.item()] for idx in top_idxs]
    scores = [prob.item() * 100 for prob in top_probs]
    return labels, scores


def draw_top3_bars(ax, labels, scores, title, color):
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, scores, color=color, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel('Уверенность (%)')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(score + 1.5, bar.get_y() + bar.get_height() / 2, f'{score:.1f}%', va='center', fontsize=10)


def main():
    file_path = choose_image()
    if not file_path:
        print('Файл не выбран.')
        return

    custom_model = SimpleCNN().to(device)
    custom_model.load_state_dict(torch.load(CUSTOM_MODEL_PATH, map_location=device))
    custom_model.eval()

    pretrained_model = build_squeezenet(num_classes=10, pretrained=False, freeze_features=False).to(device)
    pretrained_model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
    pretrained_model.eval()

    custom_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    pretrained_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    orig_image = Image.open(file_path).convert('RGB')
    custom_tensor = custom_transform(orig_image).unsqueeze(0).to(device)
    pretrained_tensor = pretrained_transform(orig_image).unsqueeze(0).to(device)

    custom_labels, custom_scores = get_top3(custom_model, custom_tensor)
    pre_labels, pre_scores = get_top3(pretrained_model, pretrained_tensor)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4.5))

    ax1.imshow(orig_image)
    ax1.set_title('Исходное изображение', fontsize=11, fontweight='bold')
    ax1.axis('off')

    draw_top3_bars(
        ax2, custom_labels, custom_scores,
        f'Кастомная CNN (ЛР 1)\n{custom_labels[0]} ({custom_scores[0]:.1f}%)',
        color='steelblue'
    )
    draw_top3_bars(
        ax3, pre_labels, pre_scores,
        f'SqueezeNet 1.1 (ЛР 2)\n{pre_labels[0]} ({pre_scores[0]:.1f}%)',
        color='seagreen'
    )

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()