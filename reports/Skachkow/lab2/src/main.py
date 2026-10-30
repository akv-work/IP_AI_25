import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import SqueezeNet1_1_Weights

from model import build_model

CLASS_NAMES = ["airplane", "bird", "car", "cat", "deer",
               "dog", "horse", "monkey", "ship", "truck"]

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build and check "
            "that NVIDIA drivers are installed. Run: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    return torch.device("cuda")

def make_loaders(data_dir: Path, batch_size: int, workers: int):
    # ImageNet normalization is required for ImageNet-pretrained weights.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = datasets.STL10(root=str(data_dir), split="train", folds=None,
                              transform=train_tf, download=True)
    test_ds = datasets.STL10(root=str(data_dir), split="test",
                             transform=test_tf, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader

def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    amp_enabled = device.type == "cuda"
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def plot_history(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, [100*x for x in history["train_acc"]], label="Train")
    axes[1].plot(epochs, [100*x for x in history["test_acc"]], label="Test")
    axes[1].set_title("Accuracy by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=160)
    plt.close(fig)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Train ImageNet-pretrained SqueezeNet 1.1 on STL-10 using CUDA.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--feature-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    device_name = torch.cuda.get_device_name(0)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print(f"Device: {device} ({device_name})")
    train_loader, test_loader = make_loaders(data_dir, args.batch_size, args.workers)

    model = build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": args.feature_lr},
        {"params": model.classifier.parameters(), "lr": args.head_lr},
    ])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = -1.0
    best_epoch = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {train_loss:.4f}, acc {train_acc*100:.2f}% | "
              f"test loss {test_loss:.4f}, acc {test_acc*100:.2f}%")
        with (out_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if test_acc > best_acc:
            best_acc, best_epoch = test_acc, epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "epoch": epoch,
                "test_accuracy": test_acc,
                "test_loss": test_loss,
            }, out_dir / "best_model.pth")
        plot_history(history, out_dir)

    elapsed = time.time() - start
    # Reload the best checkpoint and evaluate it once more for the final metrics.
    checkpoint = torch.load(out_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_loss, final_acc = run_epoch(model, test_loader, criterion, device)
    total_params = sum(p.numel() for p in model.parameters())
    metrics = {
        "device": device_name,
        "epochs_requested": args.epochs,
        "best_epoch_by_test_accuracy": best_epoch,
        "test_accuracy": final_acc,
        "test_accuracy_percent": round(final_acc * 100, 4),
        "test_loss": round(final_loss, 6),
        "training_seconds": round(elapsed, 2),
        "parameters_total": total_params,
        "test_examples": len(test_loader.dataset),
        "class_names": CLASS_NAMES,
        "optimizer": "Adam",
        "feature_lr": args.feature_lr,
        "head_lr": args.head_lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(out_dir, metrics, args, device_name)
    plot_history(history, out_dir)
    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and results to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
