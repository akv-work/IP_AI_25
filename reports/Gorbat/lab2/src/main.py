import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent
LAB1 = ROOT.parent
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
WEIGHTS = models.ResNet34_Weights.DEFAULT
TF_R = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
TF_C = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def make_resnet():
    m = models.resnet34(weights=WEIGHTS)
    m.fc = nn.Linear(m.fc.in_features, 10)
    return m


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
def predict(model, x):
    model.eval()
    p = model(x.to(DEVICE).unsqueeze(0)).softmax(1)[0]
    i = p.argmax().item()
    return CLASSES[i], p[i].item()


def load_custom():
    m = CNN().to(DEVICE)
    m.load_state_dict(torch.load(LAB1 / "cnn.pth", map_location=DEVICE, weights_only=True))
    return m


def visualize(resnet, custom, n=8):
    raw = datasets.CIFAR10(LAB1 / "data", train=False, download=True)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, i in zip(axes.flat, range(n)):
        img, y = raw[i]
        rp, rc = predict(resnet, TF_R(img))
        cp, cc = predict(custom, TF_C(img))
        ax.imshow(img)
        ax.set_title(f"true: {CLASSES[y]}\nR34: {rp} {rc:.2f}\nCNN: {cp} {cc:.2f}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "predictions.png")
    plt.close()


def predict_file(resnet, custom, path):
    img = Image.open(path).convert("RGB")
    rp, rc = predict(resnet, TF_R(img))
    cp, cc = predict(custom, TF_C(img))
    title = f"ResNet34: {rp} ({rc:.2f})  |  CNN: {cp} ({cc:.2f})"
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ROOT / "prediction.png")
    plt.close()
    print(title)


def train():
    train_ds = datasets.CIFAR10(LAB1 / "data", train=True, download=True, transform=TF_R)
    test_ds = datasets.CIFAR10(LAB1 / "data", train=False, download=True, transform=TF_R)
    train_ld = DataLoader(train_ds, 64, shuffle=True)
    test_ld = DataLoader(test_ds, 64)
    model = make_resnet().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    print("device:", DEVICE, flush=True)
    for epoch in range(3):
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
        print(f"epoch {epoch + 1}: loss={losses[-1]:.4f}", flush=True)
    acc = accuracy(model, test_ld)
    print(f"resnet34 test accuracy: {acc:.4f}", flush=True)
    custom = load_custom()
    custom_ld = DataLoader(
        datasets.CIFAR10(LAB1 / "data", train=False, download=False, transform=TF_C), 256
    )
    acc_c = accuracy(custom, custom_ld)
    print(f"custom cnn test accuracy: {acc_c:.4f}", flush=True)
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("ResNet34 training loss (CIFAR-10)")
    plt.savefig(ROOT / "loss.png")
    plt.close()
    visualize(model, custom)
    torch.save(model.state_dict(), ROOT / "resnet34.pth")
    (ROOT / "metrics.txt").write_text(
        f"resnet34={acc:.4f}\ncustom_cnn={acc_c:.4f}\n"
        f"loss={','.join(f'{v:.4f}' for v in losses)}\n"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = make_resnet().to(DEVICE)
        model.load_state_dict(torch.load(ROOT / "resnet34.pth", map_location=DEVICE, weights_only=True))
        predict_file(model, load_custom(), sys.argv[1])
    else:
        train()
