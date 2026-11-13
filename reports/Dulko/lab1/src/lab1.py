import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parents[2] / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ("airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck")
SEED = 5


class LabeledSTL10(datasets.STL10):
    train_list = datasets.STL10.train_list[:2]
    url = datasets.STL10.url.replace("http:", "https:")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for width in (32, 64, 128):
            for _ in range(2):
                layers.extend([nn.Conv2d(channels, width, 3, padding=1, bias=False),
                               nn.BatchNorm2d(width), nn.ReLU()])
                channels = width
            layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers,
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def preprocessing(transfer=False, training=False):
    size = 224 if transfer else 96
    mean = (0.485, 0.456, 0.406) if transfer else (0.5,) * 3
    std = (0.229, 0.224, 0.225) if transfer else (0.5,) * 3
    steps = [transforms.Resize(256), transforms.CenterCrop(size)] if transfer else [transforms.Resize((size, size))]
    if training:
        steps = ([transforms.Resize(256), transforms.RandomCrop(size)] if transfer
                 else [transforms.RandomCrop(size, padding=8)])
        steps.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def loaders(transfer):
    train = LabeledSTL10(DATA, split="train", download=True, transform=preprocessing(transfer, True))
    valid = LabeledSTL10(DATA, split="train", download=True, transform=preprocessing(transfer))
    test = LabeledSTL10(DATA, split="test", download=True, transform=preprocessing(transfer))
    rng = np.random.default_rng(SEED)
    train_ids, valid_ids = [], []
    for label in range(10):
        ids = rng.permutation(np.flatnonzero(train.labels == label))
        train_ids.extend(ids[:400].tolist())
        valid_ids.extend(ids[400:].tolist())
    return (DataLoader(Subset(train, train_ids), 64, shuffle=True),
            DataLoader(Subset(valid, valid_ids), 64), DataLoader(test, 64), test)


def pass_epoch(model, loader, optimizer=None):
    model.train(optimizer is not None)
    if hasattr(model, "features"):
        model.features.eval()
    loss_sum = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return loss_sum / count, 100 * correct / count


@torch.no_grad()
def classify(model, image, transfer=False):
    model.eval()
    probabilities = model(preprocessing(transfer)(image).unsqueeze(0).to(DEVICE)).softmax(1)[0]
    label = probabilities.argmax().item()
    return CLASSES[label], probabilities[label].item()


def predict(model, image_path, output, transfer=False):
    image = Image.open(image_path).convert("RGB")
    label, score = classify(model, image, transfer)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(image)
    ax.set_title(f"{label} ({score:.1%})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"{image_path}: {label} ({score:.1%})", flush=True)


def restore(model, checkpoint, transfer=False):
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    (model.classifier if transfer else model).load_state_dict(state)
    return model.to(DEVICE).eval()


def visualize(model, test, output, transfer=False):
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    indices = np.random.default_rng(SEED).choice(len(test), 8, replace=False)
    for index, ax in zip(indices, axes.flat):
        image = Image.fromarray(test.data[index].transpose(1, 2, 0))
        predicted, _ = classify(model, image, transfer)
        ax.imshow(image)
        ax.set_title(f"{predicted}\n{CLASSES[test.labels[index]]}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "predictions.png", dpi=180)
    plt.close(fig)


def train(model, output, transfer=False, epochs=60):
    output.mkdir(parents=True, exist_ok=True)
    train_ld, valid_ld, test_ld, test = loaders(transfer)
    model = model.to(DEVICE)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=0.01 if transfer else 0.05, momentum=0.9,
                                weight_decay=1e-4 if transfer else 5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    history, best_loss = [], float("inf")
    print(f"device: {DEVICE}", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, _ = pass_epoch(model, train_ld, optimizer)
        valid_loss, valid_acc = pass_epoch(model, valid_ld)
        history.append(train_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save((model.classifier if transfer else model).state_dict(), output / "model.pth")
        scheduler.step()
        print(f"epoch {epoch}: loss={train_loss:.4f}, val={valid_acc:.1f}%", flush=True)
    restore(model, output / "model.pth", transfer)
    _, test_acc = pass_epoch(model, test_ld)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(range(1, epochs + 1), history)
    ax.set(xlabel="epoch", ylabel="loss", title="Training loss (STL-10)")
    fig.tight_layout()
    fig.savefig(output / "curves.png", dpi=180)
    plt.close(fig)
    np.savetxt(output / "loss.csv", history, delimiter=",")
    visualize(model, test, output, transfer)
    print(f"test accuracy: {test_acc:.2f}%", flush=True)
    return model


if __name__ == "__main__":
    seed_everything()
    output = ROOT / "results"
    if len(sys.argv) > 1:
        model = restore(CNN(), output / "model.pth")
        predict(model, sys.argv[1], output / "custom_prediction.png")
    else:
        train(CNN(), output)
