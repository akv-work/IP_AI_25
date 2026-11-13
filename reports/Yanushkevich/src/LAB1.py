import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mnist_cnn.pt"
MEAN, STD = 0.1307, 0.3081

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((MEAN,), (STD,)),
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


def train(epochs=10, batch_size=64, lr=1.0):
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    train_losses, test_losses, test_accs = [], [], []
    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        print(f"Эпоха {epoch:2d}/{epochs} | train loss: {tr_loss:.4f} | "
              f"test loss: {te_loss:.4f} | test acc: {te_acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nИтоговая точность на тестовой выборке: {test_accs[-1]:.2f}%")

    xs = range(1, epochs + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, train_losses, marker="o", label="Обучающая выборка")
    ax[0].plot(xs, test_losses, marker="o", label="Тестовая выборка")
    ax[0].set_xlabel("Эпоха")
    ax[0].set_ylabel("Ошибка (CrossEntropyLoss)")
    ax[0].set_title("Изменение ошибки")
    ax[0].grid(True)
    ax[0].legend()
    ax[1].plot(xs, test_accs, marker="o", color="green")
    ax[1].set_xlabel("Эпоха")
    ax[1].set_ylabel("Точность, %")
    ax[1].set_title("Точность на тестовой выборке")
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=150)
    plt.show()


def load_model():
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def preprocess(img, invert=False, thicken=False):
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


def predict(image_path=None, invert=False, thicken=False):
    model = load_model()
    true_label = None

    if image_path:
        img = preprocess(Image.open(image_path), invert, thicken)
    else:
        test_ds = datasets.MNIST("./data", train=False, download=True)
        idx = np.random.randint(len(test_ds))
        img, true_label = test_ds[idx]

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).squeeze().cpu().numpy()
    pred = int(probs.argmax())

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img, cmap="gray")
    title = f"Предсказание: {pred} ({probs[pred] * 100:.1f}%)"
    if true_label is not None:
        title += f"\nИстинная метка: {true_label}"
    ax[0].set_title(title)
    ax[0].axis("off")
    ax[1].bar(range(10), probs)
    ax[1].set_xticks(range(10))
    ax[1].set_xlabel("Класс")
    ax[1].set_ylabel("Вероятность")
    ax[1].set_title("Распределение вероятностей")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "predict"])
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--thicken", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs)
    else:
        predict(args.image, args.invert, args.thicken)