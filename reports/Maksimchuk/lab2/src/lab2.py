import os
import sys
import argparse
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES  = 10
SEED         = 42
IMG_SIZE     = 224

LAB1_WEIGHTS    = "lab1_cnn_weights.pth"
LAB1_CHECKPOINT = "lab1_cnn_checkpoint.pth"
LAB2_WEIGHTS    = "lab2_mobilenet_weights.pth"
LAB2_CHECKPOINT = "lab2_mobilenet_checkpoint.pth"

os.makedirs("results", exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

fashion_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]



class FashionCNN(nn.Module):
    """Кастомная CNN из ЛР1."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 7 * 7, 512)
        self.fc2   = nn.Linear(512, NUM_CLASSES)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class MobileNetV3Fashion(nn.Module):

    def __init__(self, freeze_backbone=False):
        super().__init__()
        self.model = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

        if freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.classifier[3].parameters():
                p.requires_grad = True

    def forward(self, x):
        # Если канал 1 — повторяем до 3; если уже 3 — не трогаем
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.shape[2:] != (IMG_SIZE, IMG_SIZE):
            x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE),
                              mode="bilinear", align_corners=False)
        return self.model(x)



def get_loaders_lab1(batch_size=128):

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        "./data", train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.FashionMNIST(
        "./data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_ds, test_ds, train_loader, test_loader


def get_loaders_lab2(batch_size=64):
    """Данные для ЛР2: 224×224, 3 канала, нормализация ImageNet."""
    train_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_ds = torchvision.datasets.FashionMNIST(
        "./data", train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.FashionMNIST(
        "./data", train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_ds, test_ds, train_loader, test_loader



def evaluate(model, loader, criterion, return_preds=False):
    model.eval()
    total, correct, loss_sum = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if return_preds:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    if return_preds:
        return (loss_sum / total, 100 * correct / total,
                np.array(all_preds), np.array(all_labels))
    return loss_sum / total, 100 * correct / total


def train_model(model, train_loader, test_loader, train_size,
                num_epochs, lr, weights_path, checkpoint_path,
                meta, args_load, args_train, history=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    train_losses = history.get('train_losses', []) if history else []
    test_losses  = history.get('test_losses',  []) if history else []
    test_accs    = history.get('test_accs',    []) if history else []
    start_epoch  = history.get('epoch', -1) + 1 if history else 0

    if (not args_load) or args_train:
        print(f"\nОбучение: {num_epochs} эпох\n")
        best_acc = max(test_accs) if test_accs else 0.0
        last_epoch = start_epoch - 1

        for epoch in range(start_epoch, num_epochs):
            last_epoch = epoch
            model.train()
            running_loss = 0
            t0 = time.time()

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / train_size
            test_loss, test_acc = evaluate(model, test_loader, criterion)

            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accs.append(test_acc)

            print(f"Epoch {epoch+1:>3}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}% | "
                  f"{time.time()-t0:.1f}s")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), weights_path)
                print(f"   ↑ Best: {best_acc:.2f}% → {weights_path}")

        torch.save(model.state_dict(), weights_path)
        torch.save({
            'epoch': last_epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'test_losses':  test_losses,
            'test_accs':    test_accs,
            'best_acc':     best_acc,
            **meta,
        }, checkpoint_path)

        print(f"\n✅ Сохранено:")
        print(f"   • {weights_path}")
        print(f"   • {checkpoint_path}")

    return train_losses, test_losses, test_accs



def run_lab1(args):
    print("=" * 70)
    print("ЛР № 1 — Кастомная CNN на Fashion-MNIST")
    print("=" * 70)

    train_ds, test_ds, train_loader, test_loader = get_loaders_lab1()
    model = FashionCNN().to(DEVICE)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}")

    history = None
    if args.load and os.path.exists(LAB1_CHECKPOINT):
        print(f"→ Загрузка чекпоинта: {LAB1_CHECKPOINT}")
        history = torch.load(LAB1_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(history['model_state_dict'])
    elif args.load and os.path.exists(LAB1_WEIGHTS):
        print(f"→ Загрузка весов: {LAB1_WEIGHTS}")
        model.load_state_dict(torch.load(LAB1_WEIGHTS, map_location=DEVICE))

    meta = {
        'classes':      fashion_classes,
        'architecture': 'FashionCNN (custom)',
        'optimizer':    'Adadelta',
        'dataset':      'Fashion-MNIST',
    }

    train_losses, test_losses, test_accs = train_model(
        model, train_loader, test_loader, len(train_ds),
        args.epochs, lr=1.0,
        weights_path=LAB1_WEIGHTS,
        checkpoint_path=LAB1_CHECKPOINT,
        meta=meta, args_load=args.load, args_train=args.train,
        history=history,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nФинальная точность ЛР1: {test_acc:.2f}%")

    if train_losses:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses,  label="Test Loss")
        plt.legend(); plt.title("Loss ЛР1"); plt.xlabel("Epoch")
        plt.subplot(1, 2, 2)
        plt.plot(test_accs, label="Test Accuracy", color='blue')
        plt.legend(); plt.title("Accuracy ЛР1"); plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig("results/lab1_curves.png", dpi=120)
        plt.close()
        print("📊 results/lab1_curves.png")



def run_lab2(args):
    print("=" * 70)
    print("ЛР № 2 (Вариант 12) — MobileNetV3-Small + Adadelta")
    print("=" * 70)

    train_ds, test_ds, train_loader, test_loader = get_loaders_lab2()
    model = MobileNetV3Fashion(freeze_backbone=args.freeze).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Обучаемых параметров: {trainable:,} / {total:,}")

    history = None
    if args.load and os.path.exists(LAB2_CHECKPOINT):
        print(f"→ Загрузка чекпоинта: {LAB2_CHECKPOINT}")
        history = torch.load(LAB2_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(history['model_state_dict'])
    elif args.load and os.path.exists(LAB2_WEIGHTS):
        print(f"→ Загрузка весов: {LAB2_WEIGHTS}")
        model.load_state_dict(torch.load(LAB2_WEIGHTS, map_location=DEVICE))

    meta = {
        'classes':      fashion_classes,
        'architecture': 'MobileNetV3-Small',
        'optimizer':    'Adadelta',
        'dataset':      'Fashion-MNIST',
    }

    train_losses, test_losses, test_accs = train_model(
        model, train_loader, test_loader, len(train_ds),
        args.epochs, lr=1.0,
        weights_path=LAB2_WEIGHTS,
        checkpoint_path=LAB2_CHECKPOINT,
        meta=meta, args_load=args.load, args_train=args.train,
        history=history,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion, return_preds=True)
    print(f"\nФинальная точность ЛР2: {test_acc:.2f}%")

    if train_losses:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(test_losses,  label="Test Loss")
        plt.legend(); plt.title("Loss ЛР2 (MobileNetV3)"); plt.xlabel("Epoch")
        plt.subplot(1, 2, 2)
        plt.plot(test_accs, label="Test Accuracy", color='green')
        plt.legend(); plt.title("Accuracy ЛР2"); plt.xlabel("Epoch")
        plt.tight_layout()
        plt.savefig("results/lab2_curves.png", dpi=120)
        plt.close()
        print("📊 results/lab2_curves.png")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=fashion_classes, yticklabels=fashion_classes)
    plt.title("Confusion Matrix — MobileNetV3 + Adadelta")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("results/lab2_cm.png", dpi=120)
    plt.close()
    print("📊 results/lab2_cm.png")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred,
          target_names=fashion_classes, zero_division=0))



def run_compare(args):
    print("=" * 70)
    print("СРАВНЕНИЕ ЛР1 vs ЛР2")
    print("=" * 70)

    if not os.path.exists(LAB1_WEIGHTS):
        print(f"❌ Не найдены веса ЛР1: {LAB1_WEIGHTS}")
        print("   Запусти: python main.py lab1")
        return
    if not os.path.exists(LAB2_WEIGHTS):
        print(f"❌ Не найдены веса ЛР2: {LAB2_WEIGHTS}")
        print("   Запусти: python main.py lab2")
        return


    _, _, _, loader1 = get_loaders_lab1(batch_size=128)
    _, _, _, loader2 = get_loaders_lab2(batch_size=64)

    m1 = FashionCNN().to(DEVICE)
    m1.load_state_dict(torch.load(LAB1_WEIGHTS, map_location=DEVICE))

    m2 = MobileNetV3Fashion().to(DEVICE)
    m2.load_state_dict(torch.load(LAB2_WEIGHTS, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    loss1, acc1 = evaluate(m1, loader1, criterion)
    loss2, acc2 = evaluate(m2, loader2, criterion)

    print("\n" + "=" * 55)
    print(f"{'Модель':<25} {'Loss':<12} {'Accuracy (%)':<12}")
    print("-" * 55)
    print(f"{'ЛР1: Custom CNN':<25} {loss1:<12.4f} {acc1:<12.2f}")
    print(f"{'ЛР2: MobileNetV3':<25} {loss2:<12.4f} {acc2:<12.2f}")
    print("-" * 55)
    print(f"Δ Accuracy: {acc2 - acc1:+.2f} п.п.")
    print("=" * 55)



def run_visualize(args):
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ: ЛР1 vs ЛР2")
    print("=" * 70)

    if not os.path.exists(LAB1_WEIGHTS) or not os.path.exists(LAB2_WEIGHTS):
        print("❌ Сначала обучи обе модели:")
        print("   python main.py lab1 && python main.py lab2")
        return

    m1 = FashionCNN().to(DEVICE)
    m1.load_state_dict(torch.load(LAB1_WEIGHTS, map_location=DEVICE))
    m1.eval()

    m2 = MobileNetV3Fashion().to(DEVICE)
    m2.load_state_dict(torch.load(LAB2_WEIGHTS, map_location=DEVICE))
    m2.eval()


    raw_ds = torchvision.datasets.FashionMNIST(
        "./data", train=False, download=True,
        transform=transforms.ToTensor())


    tf_lab1 = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])

    tf_lab2 = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    N = 5
    idxs = random.sample(range(len(raw_ds)), N)

    fig, axes = plt.subplots(2, N, figsize=(3 * N, 7))
    for col, idx in enumerate(idxs):
        img_raw, true_label = raw_ds[idx]

        inp1 = tf_lab1(img_raw).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p1 = F.softmax(m1(inp1), dim=1)[0]
            pred1 = p1.argmax().item()
            conf1 = p1.max().item() * 100

        inp2 = tf_lab2(img_raw).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p2 = F.softmax(m2(inp2), dim=1)[0]
            pred2 = p2.argmax().item()
            conf2 = p2.max().item() * 100

        axes[0, col].imshow(img_raw.squeeze(), cmap="gray")
        axes[0, col].set_title(
            f"ЛР1: {fashion_classes[pred1]}\n({conf1:.1f}%)"
            f"{'✅' if pred1 == true_label else '❌'}",
            fontsize=9,
            color='green' if pred1 == true_label else 'red')
        axes[0, col].axis("off")

        axes[1, col].imshow(img_raw.squeeze(), cmap="gray")
        axes[1, col].set_title(
            f"ЛР2: {fashion_classes[pred2]}\n({conf2:.1f}%)"
            f"{'✅' if pred2 == true_label else '❌'}",
            fontsize=9,
            color='green' if pred2 == true_label else 'red')
        axes[1, col].axis("off")

        print(f"[{col+1}] True: {fashion_classes[true_label]:<12} | "
              f"ЛР1: {fashion_classes[pred1]:<12} ({conf1:.1f}%) | "
              f"ЛР2: {fashion_classes[pred2]:<12} ({conf2:.1f}%)")

    plt.suptitle("Сравнение ЛР1 (Custom CNN) vs ЛР2 (MobileNetV3)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("results/visualization_comparison.png", dpi=120)
    plt.close()
    print("\n📊 results/visualization_comparison.png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', nargs='?', default='all',
                        choices=['lab1', 'lab2', 'compare',
                                 'visualize', 'all'],
                        help='Что запустить (default: all)')
    parser.add_argument('--load',   action='store_true', help='Загрузить веса')
    parser.add_argument('--train',  action='store_true', help='Дообучить')
    parser.add_argument('--freeze', action='store_true',
                        help='Заморозить backbone (только ЛР2)')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Число эпох')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.command == 'lab1':
        run_lab1(args)
    elif args.command == 'lab2':
        if args.epochs == 25:
            args.epochs = 15
        run_lab2(args)
    elif args.command == 'compare':
        run_compare(args)
    elif args.command == 'visualize':
        run_visualize(args)
    elif args.command == 'all':
        run_lab1(args)
        args.epochs = 15
        run_lab2(args)
        run_compare(args)
        run_visualize(args)