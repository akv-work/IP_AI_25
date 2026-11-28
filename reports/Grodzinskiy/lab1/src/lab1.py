import sys

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

CLASSES = datasets.CIFAR100("data", train=True, download=True).classes

TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        return self.net(x)


def loaders():
    train = datasets.CIFAR100("data", train=True, download=True, transform=TF)
    test = datasets.CIFAR100("data", train=False, download=True, transform=TF)
    return DataLoader(train, 128, shuffle=True), DataLoader(test, 256), test


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    ok = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        ok += (model(x).argmax(1) == y).sum().item()
        n += y.size(0)
    return ok / n


@torch.no_grad()
def predict_tensor(model, x):
    model.eval()
    p = model(x.to(DEVICE).unsqueeze(0)).softmax(1)[0]
    i = p.argmax().item()
    return CLASSES[i], p[i].item()


def show_image(img, title, path):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def predict_file(model, path):
    img = Image.open(path).convert("RGB").resize((32, 32))
    label, conf = predict_tensor(model, TF(img))
    title = f"{label} ({conf:.2f})"
    show_image(img, title, "prediction.png")
    print(title)


def visualize_samples(model, test, n=8):
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for ax, i in zip(axes.flat, range(n)):
        x, y = test[i]
        pred, conf = predict_tensor(model, x)
        ax.imshow((x * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1))
        ax.set_title(f"{pred}\n{CLASSES[y]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("predictions.png")
    plt.close()


def train():
    train_ld, test_ld, test = loaders()
    model = CNN().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    print("device:", DEVICE)

    for epoch in range(10):
        model.train()
        total = 0
        for x, y in train_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        losses.append(total / len(train_ld))
        print(f"epoch {epoch + 1}: loss={losses[-1]:.4f}")

    acc = accuracy(model, test_ld)
    print(f"test accuracy: {acc:.4f}")

    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss (CIFAR-100)")
    plt.savefig("loss_cifar100.png")
    plt.close()

    visualize_samples(model, test)
    torch.save(model.state_dict(), "cnn_cifar100.pth")

    with open("metrics_cifar100.txt", "w") as f:
        f.write(f"test_accuracy={acc:.4f}\n")
        f.write("loss=" + ",".join(f"{v:.4f}" for v in losses) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = CNN().to(DEVICE)
        model.load_state_dict(
            torch.load("cnn_cifar100.pth", map_location=DEVICE, weights_only=True)
        )
        predict_file(model, sys.argv[1])
    else:
        train()