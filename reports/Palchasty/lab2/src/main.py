import random
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


CONFIG = {
    "seed": 42,
    "classes": ('airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'),
    "mean_cifar": (0.4914, 0.4822, 0.4465),
    "std_cifar":  (0.2470, 0.2435, 0.2616),
    "mean_imagenet": (0.485, 0.456, 0.406),
    "std_imagenet":  (0.229, 0.224, 0.225),
    "bs_scratch": 128,
    "bs_finetune": 64,
    "epochs_scratch": 10,
    "epochs_finetune": 5,
    "lr": 1e-3,
}

CLASSES = CONFIG["classes"]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(batch_train, batch_test, resize_to=None, mean=None, std=None):
    ops_train = [T.Resize((resize_to, resize_to))] if resize_to else [T.RandomCrop(32, padding=4)]
    ops_train += [T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)]

    ops_test = [T.Resize((resize_to, resize_to))] if resize_to else []
    ops_test += [T.ToTensor(), T.Normalize(mean, std)]

    tr = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=T.Compose(ops_train))
    ts = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=T.Compose(ops_test))
    return (tr, ts,
            DataLoader(tr, batch_size=batch_train, shuffle=True, num_workers=2),
            DataLoader(ts, batch_size=batch_test, shuffle=False, num_workers=2))


class SmallConvNet(nn.Module):
    def __init__(self, n_out=10, p_drop=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_out),
        )

    def forward(self, x):
        return self.head(self.features(x))


class Trainer:
    def __init__(self, net, lr=1e-3):
        self.net = net.to(DEVICE)
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        self.log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    def _step_epoch(self, loader, train_flag):
        self.net.train() if train_flag else self.net.eval()
        run_loss, hits, seen = 0.0, 0, 0
        ctx = torch.enable_grad() if train_flag else torch.no_grad()
        with ctx:
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if train_flag:
                    self.opt.zero_grad()
                logits = self.net(xb)
                loss = self.loss_fn(logits, yb)
                if train_flag:
                    loss.backward()
                    self.opt.step()
                run_loss += loss.item() * xb.size(0)
                hits += (logits.argmax(1) == yb).sum().item()
                seen += yb.size(0)
        return run_loss / seen, hits / seen

    def fit(self, tr_loader, ts_loader, n_epochs):
        for e in range(1, n_epochs + 1):
            l_tr, a_tr = self._step_epoch(tr_loader, True)
            l_ts, a_ts = self._step_epoch(ts_loader, False)
            for k, v in zip(("train_loss", "train_acc"),
                            (l_tr, a_tr)):        self.log[k].append(v)
            for k, v in zip(("test_loss", "test_acc"),
                            (l_ts, a_ts)):        self.log[k].append(v)
            print(f"[{e:2d}/{n_epochs}]  tr_loss={l_tr:.4f}  tr_acc={a_tr*100:5.2f}%"
                  f"   |   ts_loss={l_ts:.4f}  ts_acc={a_ts*100:5.2f}%")
        return self.log


def plot_history(log, tag):
    rng = range(1, len(log["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(rng, log["train_loss"], 'o-', ms=3, label="train")
    ax[0].plot(rng, log["test_loss"],  'o-', ms=3, label="test")
    ax[0].set(xlabel="эпоха", ylabel="loss", title=f"{tag}: функция потерь")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(rng, np.array(log["train_acc"]) * 100, 'o-', ms=3, label="train")
    ax[1].plot(rng, np.array(log["test_acc"])  * 100, 'o-', ms=3, label="test")
    ax[1].set(xlabel="эпоха", ylabel="accuracy, %", title=f"{tag}: точность")
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()


def predict_pair(img_path, scratch_model, ft_model):
    img = Image.open(img_path).convert("RGB")

    prep_scratch = T.Compose([T.Resize((32, 32)), T.ToTensor(),
                              T.Normalize(CONFIG["mean_cifar"], CONFIG["std_cifar"])])
    prep_ft = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                         T.Normalize(CONFIG["mean_imagenet"], CONFIG["std_imagenet"])])

    scratch_model.eval(); ft_model.eval()
    with torch.no_grad():
        p_s = F.softmax(scratch_model(prep_scratch(img).unsqueeze(0).to(DEVICE)), 1).cpu().numpy().ravel()
        p_f = F.softmax(ft_model(prep_ft(img).unsqueeze(0).to(DEVICE)), 1).cpu().numpy().ravel()

    idx_s, idx_f = int(p_s.argmax()), int(p_f.argmax())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1.2, 1.2]})
    ax[0].imshow(img); ax[0].axis("off"); ax[0].set_title("что подаём на вход")

    ax[1].barh(CLASSES, p_s * 100,
               color=["crimson" if i == idx_s else "steelblue" for i in range(len(CLASSES))])
    ax[1].invert_yaxis(); ax[1].set_xlim(0, 100)
    ax[1].set_title(f"SmallConvNet → {CLASSES[idx_s]}"); ax[1].set_xlabel("вероятность, %")

    ax[2].barh(CLASSES, p_f * 100,
               color=["seagreen" if i == idx_f else "steelblue" for i in range(len(CLASSES))])
    ax[2].invert_yaxis(); ax[2].set_xlim(0, 100)
    ax[2].set_title(f"MobileNetV3 → {CLASSES[idx_f]}"); ax[2].set_xlabel("вероятность, %")

    plt.tight_layout(); plt.show()
    return idx_s, idx_f


def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    print("Устройство для вычислений:", DEVICE)


    train_set, test_set, train_loader, test_loader = make_loaders(
        CONFIG["bs_scratch"], 256,
        mean=CONFIG["mean_cifar"], std=CONFIG["std_cifar"])

    print(f"Обучающих изображений: {len(train_set)} | тестовых: {len(test_set)}")
    print("Размер тензора одной картинки:", train_set[0][0].shape)


    scratch_net = SmallConvNet(n_out=len(CLASSES))
    scratch_params = sum(p.numel() for p in scratch_net.parameters() if p.requires_grad)
    print(scratch_net)
    print(f"Параметров в самодельной сети: {scratch_params:,}")

    scratch_trainer = Trainer(scratch_net, lr=CONFIG["lr"])
    hist_scratch = scratch_trainer.fit(train_loader, test_loader, CONFIG["epochs_scratch"])

    scratch_model = scratch_trainer.net


    train_set_pt, test_set_pt, train_loader_pt, test_loader_pt = make_loaders(
        CONFIG["bs_finetune"], 128, resize_to=224,
        mean=CONFIG["mean_imagenet"], std=CONFIG["std_imagenet"])[0:4]

    print(f"Датасет для finetune: {len(train_set_pt)} / {len(test_set_pt)}")
    print("Размер входа после ресайза:", train_set_pt[0][0].shape)


    backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in backbone.features.parameters():
        p.requires_grad = False

    n_feat = backbone.classifier[-1].in_features
    backbone.classifier[-1] = nn.Linear(n_feat, len(CLASSES))
    backbone = backbone.to(DEVICE)

    total_p = sum(p.numel() for p in backbone.parameters())
    train_p = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print("Новый классификатор:", backbone.classifier)
    print(f"Всего весов: {total_p:,} | обучаемых (только голова): {train_p:,}")

    ft_trainer = Trainer(backbone, lr=CONFIG["lr"])
    hist_ft = ft_trainer.fit(train_loader_pt, test_loader_pt, CONFIG["epochs_finetune"])
    ft_model = ft_trainer.net


    plot_history(hist_ft, "MobileNetV3")
    print(f"Итог MobileNetV3 на тесте: {hist_ft['test_acc'][-1]*100:.2f}%")

    print("\n--- Сводка ---")
    print(f"SmallConvNet : {hist_scratch['test_acc'][-1]*100:.2f}%  "
          f"({scratch_params:,} параметров, все обучаются)")
    print(f"MobileNetV3  : {hist_ft['test_acc'][-1]*100:.2f}%  "
          f"({train_p:,} обучаемых из {total_p:,})")


    benchmarks = {
        "PyTorch tutorial (простая модель)": 54.0,
        "Keras baseline": 67.0,
        "Лучший результат 2011 г.": 80.5,
        "ImageNet-признаки + классификатор (2015)": 89.14,
        "ResNet / DenseNet": 92.5,
        "Человек": 94.0,
        "ViT (>95%)": 95.0,
        "SOTA 2026: ViT-H/14 (JFT-300M)": 99.5,
    }
    mine = {
        "SmallConvNet": hist_scratch['test_acc'][-1] * 100,
        "MobileNetV3":  hist_ft['test_acc'][-1] * 100,
    }
    merged = {**benchmarks, **mine}

    order = sorted(merged, key=merged.get)
    vals  = [merged[k] for k in order]
    palette = ["tab:orange" if k in mine else "lightgray" for k in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(order, vals, color=palette)
    for b, v in zip(bars, vals):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center")
    ax.set_xlim(40, 105)
    ax.set_xlabel("Accuracy на CIFAR-10, %")
    ax.set_title("Наши модели (оранжевые) среди известных результатов")
    ax.grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.show()

    acc_scratch, acc_ft = mine["SmallConvNet"], mine["MobileNetV3"]
    SOTA, HUMAN = 99.5, 94.0
    print("\n--- Отставание от ориентиров ---")
    print(f"SmallConvNet: {acc_scratch:.2f}%   "
          f"| до человека: {HUMAN - acc_scratch:.2f} п.п.  | до SOTA: {SOTA - acc_scratch:.2f} п.п.")
    print(f"MobileNetV3 : {acc_ft:.2f}%   "
          f"| до человека: {HUMAN - acc_ft:.2f} п.п.  | до SOTA: {SOTA - acc_ft:.2f} п.п.")
    print(f"Выигрыш MobileNetV3 над своей CNN: {acc_ft - acc_scratch:+.2f} п.п.")


    predict_pair(r'C:\Users\arosl\PycharmProjects\OIS2\bird.jpeg',
                 scratch_model, ft_model)


if __name__ == '__main__':

    from multiprocessing import freeze_support
    freeze_support()
    main()