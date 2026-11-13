import random, ssl, sys, certifi, torch
from torch import nn, optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
dev = torch.device("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
DATA = Path(__file__).resolve().parents[1] / "data"

# Стандарты нормализации для предобученных сетей torchvision (ImageNet)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Аугментация / предобработка для DenseNet121 (требует размер 224x224 и 3 канала)
aug = transforms.Compose([
    transforms.Resize(224), 
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Предобработка для кастомной CNN (CIFAR-100 уже имеет размер 32x32 и 3 канала)
small = transforms.Compose([
    transforms.Resize(32), 
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_loaders(tf, batch):
    tr = datasets.CIFAR100(DATA, train=True, download=True, transform=tf)
    te = datasets.CIFAR100(DATA, train=False, download=True, transform=tf)
    return DataLoader(tr, batch, shuffle=True), DataLoader(te, batch, shuffle=False)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), 
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(), 
            nn.Linear(256, 100)  # 100 классов для CIFAR-100
        )
    def forward(self, x): return self.body(x)

def train(model, loader, opt, epochs, tag):
    crit, hist = nn.CrossEntropyLoss(), []
    for ep in range(epochs):
        model.train(); total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            loss = crit(model(imgs), labels)
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item()
        avg = total / len(loader); hist.append(avg)
        print(f"[{tag}] эпоха {ep+1}: loss = {avg:.4f}")
    return hist

def accuracy(model, loader):
    model.eval(); correct = seen = 0
    with torch.no_grad():
        for imgs, labels in loader:
            correct += (model(imgs.to(dev)).argmax(1).cpu() == labels).sum().item()
            seen += labels.numel()
    return 100 * correct / seen

# 1. Обучение кастомной CNN (ЛР1) с оптимизатором Adam
cnn = SmallCNN().to(dev)
tr_c, te_c = get_loaders(small, 64)
hist_cnn = train(cnn, tr_c, optim.Adam(cnn.parameters(), lr=0.001), 3, "CNN")
acc_cnn = accuracy(cnn, te_c)

# 2. Загрузка и настройка DenseNet121 для Варианта 9 (ЛР2)
densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
for p in densenet.features.parameters(): 
    p.requires_regul = False # Замораживаем базовые слои сверток
    p.requires_grad = False

# Заменяем классификатор под 100 классов CIFAR-100
densenet.classifier = nn.Linear(densenet.classifier.in_features, 100)
densenet = densenet.to(dev)

tr_d, te_d = get_loaders(aug, 32)
# Обучение с тем же оптимизатором (Adam) для классификатора
hist_densenet = train(densenet, tr_d, optim.Adam(densenet.classifier.parameters(), lr=0.001), 2, "DenseNet121")
acc_densenet = accuracy(densenet, te_d)

print(f"\nCNN (ЛР1): {acc_cnn:.2f}%    DenseNet121 (ЛР2): {acc_densenet:.2f}%")

# График обучения
plt.figure()
plt.plot(range(1, len(hist_cnn)+1), hist_cnn, marker="o", label="CNN")
plt.plot(range(1, len(hist_densenet)+1), hist_densenet, marker="s", label="DenseNet121")
plt.xlabel("Эпоха"); plt.ylabel("CrossEntropyLoss")
plt.title("Ошибка на обучении (CIFAR-100, Вариант 9)")
plt.legend(); plt.grid(True)

# Визуализация работы на изображении
raw = datasets.CIFAR100(DATA, train=False)
if len(sys.argv) > 1:
    img, truth = Image.open(sys.argv[1]).convert("RGB"), None
else:
    img, truth = raw[random.randrange(len(raw))]

with torch.no_grad():
    pred_cnn     = cnn(small(img).unsqueeze(0).to(dev)).argmax(1).item()
    pred_densenet = densenet(aug(img).unsqueeze(0).to(dev)).argmax(1).item()

plt.figure(); plt.imshow(img); plt.axis("off")
plt.title(f"CNN={pred_cnn}, DenseNet={pred_densenet}"
          + ("" if truth is None else f", истина={truth}"))
plt.show()