import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_PATH = "resnet34_mnist.pt"
CUSTOM_PATH = "mnist_cnn.pt"
IMG_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

resnet_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


def build_resnet34(pretrained=True):
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        total_loss += criterion(output, target).item() * data.size(0)
        correct += (output.argmax(1) == target).sum().item()
    n = len(loader.dataset)
    return total_loss / n, 100.0 * correct / n


def train(epochs=3, batch_size=128, lr=1.0):
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=resnet_transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=resnet_transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=500, num_workers=2)

    model = build_resnet34(pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    train_losses, test_losses, test_accs = [], [], []
    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        print(f"Эпоха {epoch:2d}/{epochs} | train loss: {tr_loss:.4f} | "
              f"test loss: {te_loss:.4f} | test acc: {te_acc:.2f}%")
    train_time = time.time() - start

    torch.save(model.state_dict(), RESNET_PATH)
    print(f"\nResNet34: итоговая точность на тесте {test_accs[-1]:.2f}%, время обучения {train_time:.0f} с")

    xs = range(1, epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, train_losses, marker="o", label="Обучающая выборка")
    ax[0].plot(xs, test_losses, marker="o", label="Тестовая выборка")
    ax[0].set_xlabel("Эпоха")
    ax[0].set_ylabel("Ошибка (CrossEntropyLoss)")
    ax[0].set_title("ResNet34: изменение ошибки")
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(xs, test_accs, marker="o", color="green")
    ax[1].set_xlabel("Эпоха")
    ax[1].set_ylabel("Точность, %")
    ax[1].set_title("ResNet34: точность на тесте")
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot_resnet34.png", dpi=150)
    plt.show()

    compare(resnet_acc=test_accs[-1], resnet_time=train_time)


def load_resnet():
    model = build_resnet34(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
    model.eval()
    return model


def load_custom():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(CUSTOM_PATH, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def timed_eval(model, transform):
    ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=500, num_workers=2)
    start = time.time()
    loss, acc = evaluate(model, loader, nn.CrossEntropyLoss())
    return loss, acc, time.time() - start


def compare(resnet_acc=None, resnet_time=None):
    rows = []
    try:
        custom = load_custom()
        loss, acc, t = timed_eval(custom, custom_transform)
        rows.append(("Кастомная СНС (ЛР1)", count_params(custom), acc, loss, t))
    except FileNotFoundError:
        print(f"Файл {CUSTOM_PATH} не найден: сначала обучите модель из ЛР1")
    try:
        resnet = load_resnet()
        loss, acc, t = timed_eval(resnet, resnet_transform)
        rows.append(("ResNet34 (предобученная)", count_params(resnet), acc, loss, t))
    except FileNotFoundError:
        print(f"Файл {RESNET_PATH} не найден: сначала выполните train")

    print("\n{:<28}{:>14}{:>12}{:>12}{:>16}".format("Модель", "Параметры", "Точность,%", "Loss", "Инференс, с"))
    for name, params, acc, loss, t in rows:
        print("{:<28}{:>14,}{:>12.2f}{:>12.4f}{:>16.1f}".format(name, params, acc, loss, t))

    if rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([r[0] for r in rows], [r[2] for r in rows], color=["tab:blue", "tab:orange"][:len(rows)])
        lo = min(r[2] for r in rows)
        ax.set_ylim(max(0, lo - 1.5), 100)
        ax.set_ylabel("Точность на тесте, %")
        ax.set_title("Сравнение моделей на MNIST")
        for i, r in enumerate(rows):
            ax.text(i, r[2] + 0.05, f"{r[2]:.2f}%", ha="center")
        plt.tight_layout()
        plt.savefig("compare_models.png", dpi=150)
        plt.show()


def preprocess_custom_image(img, invert=False, thicken=False):
    img = img.convert("L")
    if invert:
        img = ImageOps.invert(img)
    arr = np.array(img)
    ys, xs = np.where(arr > arr.max() * 0.2)
    img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = img.size
    if thicken:
        k = max(3, (max(w, h) // 15) | 1)
        img = img.filter(ImageFilter.MaxFilter(k))
    scale = 20 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2))
    return canvas


@torch.no_grad()
def probs_of(model, img, transform):
    x = transform(img).unsqueeze(0).to(DEVICE)
    return F.softmax(model(x), dim=1).squeeze().cpu().numpy()


def predict(image_path=None, invert=False, thicken=False):
    resnet = load_resnet()
    custom = load_custom()
    true_label = None

    if image_path:
        img = preprocess_custom_image(Image.open(image_path), invert, thicken)
    else:
        test_ds = datasets.MNIST("./data", train=False, download=True)
        img, true_label = test_ds[np.random.randint(len(test_ds))]

    p_res = probs_of(resnet, img, resnet_transform)
    p_cus = probs_of(custom, img, custom_transform)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].imshow(img, cmap="gray")
    title = "Входное изображение"
    if true_label is not None:
        title += f"\nИстинная метка: {true_label}"
    ax[0].set_title(title)
    ax[0].axis("off")
    for a, p, name, color in ((ax[1], p_res, "ResNet34", "tab:orange"), (ax[2], p_cus, "Кастомная СНС", "tab:blue")):
        a.bar(range(10), p, color=color)
        a.set_xticks(range(10))
        a.set_ylim(0, 1)
        a.set_xlabel("Класс")
        a.set_ylabel("Вероятность")
        a.set_title(f"{name}: {int(p.argmax())} ({p.max() * 100:.1f}%)")
    plt.tight_layout()
    plt.savefig("predict_result.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "predict", "compare"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--thicken", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs)
    elif args.mode == "compare":
        compare()
    else:
        predict(args.image, args.invert, args.thicken)