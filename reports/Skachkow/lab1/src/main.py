import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
USE_AMP = DEVICE.type == "cuda"
EPOCHS = 60
BATCH = 128
MAX_LR = 3e-3
WEIGHTS = "cnn_cifar100_v2.pt"

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        def block(cin, cout, p):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(p),
            )

        self.features = nn.Sequential(
            block(3, 128, 0.0),     # 32 -> 16
            block(128, 256, 0.1),   # 16 -> 8
            block(256, 512, 0.2),   # 8  -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_loaders():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    train_set = torchvision.datasets.CIFAR100("./data", train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR100("./data", train=False, download=True, transform=test_tf)

    kw = dict(num_workers=4, pin_memory=(DEVICE.type == "cuda"), persistent_workers=True)
    train_loader = torch.utils.data.DataLoader(train_set, BATCH, shuffle=True, **kw)
    test_loader = torch.utils.data.DataLoader(test_set, 256, shuffle=False, **kw)
    return train_loader, test_loader, train_set.classes


def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.autocast(DEVICE.type, enabled=USE_AMP):
                out = model(x)
            loss_sum += criterion(out.float(), y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return loss_sum / total, correct / total


def train():
    train_loader, test_loader, _ = get_loaders()
    model = SimpleCNN().to(DEVICE)

    train_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss()  # обычная CE для честного графика

    optimizer = optim.Adam(model.parameters(), lr=MAX_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(train_loader)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    hist = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, enabled=USE_AMP):
                out = model(x)
                loss = train_criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

        tr_loss, tr_acc = loss_sum / total, correct / total
        te_loss, te_acc = evaluate(model, test_loader, eval_criterion)
        for k, v in zip(hist, (tr_loss, te_loss, tr_acc, te_acc)):
            hist[k].append(v)
        print(f"Эпоха {epoch:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"test loss {te_loss:.3f} acc {te_acc:.3f}", flush=True)

    torch.save(model.state_dict(), WEIGHTS)
    print(f"\nИтоговая точность на тестовой выборке: {hist['test_acc'][-1] * 100:.2f}%")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["train_loss"], label="train (с label smoothing)")
    ax[0].plot(hist["test_loss"], label="test")
    ax[0].set_title("Ошибка (CrossEntropyLoss)")
    ax[0].set_xlabel("Эпоха"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(hist["train_acc"], label="train")
    ax[1].plot(hist["test_acc"], label="test")
    ax[1].set_title("Точность (accuracy)")
    ax[1].set_xlabel("Эпоха"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("training_curves_v2.png", dpi=150)
    plt.show()


def predict(path):
    classes = torchvision.datasets.CIFAR100("./data", train=False, download=True).classes
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS, map_location=DEVICE))
    model.eval()

    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(MEAN, STD)])
    x = tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x).float(), dim=1)[0]
    top_p, top_i = probs.topk(5)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img); ax[0].axis("off")
    ax[0].set_title(f"Результат: {classes[top_i[0]]} ({top_p[0] * 100:.1f}%)")
    ax[1].barh([classes[i] for i in top_i.flip(0)], top_p.flip(0).cpu() * 100)
    ax[1].set_xlabel("Вероятность, %"); ax[1].set_title("Топ-5 классов")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "predict":
        predict(sys.argv[2])
    else:
        train()