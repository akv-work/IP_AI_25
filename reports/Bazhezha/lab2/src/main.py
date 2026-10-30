import random, ssl, sys, certifi, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib.pyplot as plt

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DATA = Path(__file__).resolve().parents[1] / "data"
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf_a = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), norm])
tf_c = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])

def loaders(tf, bs):
    return (DataLoader(datasets.MNIST(DATA, train=True, download=True, transform=tf), bs, True),
            DataLoader(datasets.MNIST(DATA, train=False, download=True, transform=tf), bs))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x)

def fit(model, ld, opt, epochs, name):
    crit, hist = nn.CrossEntropyLoss(), []
    for e in range(epochs):
        model.train(); s = 0
        for x, y in ld:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); s += loss.item()
        hist.append(s / len(ld)); print(f"{name} epoch {e + 1} loss {hist[-1]:.4f}")
    return hist

def acc(model, ld):
    model.eval(); ok = n = 0
    with torch.no_grad():
        for x, y in ld:
            ok += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return 100 * ok / n

def to_digit(pil):
    t = transforms.ToTensor()(pil.convert("L"))
    return transforms.ToPILImage()(1 - t if t.mean() > 0.5 else t)

cnn = CNN().to(device)
tr_c, te_c = loaders(tf_c, 64)
loss_c = fit(cnn, tr_c, torch.optim.SGD(cnn.parameters(), 0.01, 0.9), 3, "CNN")
acc_c = acc(cnn, te_c)

alex = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
for p in alex.features.parameters():
    p.requires_grad = False
alex.classifier[6] = nn.Linear(4096, 10)
alex = alex.to(device)
tr_a, te_a = loaders(tf_a, 32)
loss_a = fit(alex, tr_a, torch.optim.SGD(alex.classifier.parameters(), 0.01, 0.9), 2, "AlexNet")
acc_a = acc(alex, te_a)

print(f"CNN (ЛР1): {acc_c:.2f}%  |  AlexNet (ЛР2): {acc_a:.2f}%")
print("SOTA MNIST ≈ 99.8% (CLoVE / ансамбли CNN).")
print("Вывод: кастомная СНС на 28×28 обычно сопоставима или лучше transfer-AlexNet,")
print("т.к. ImageNet-признаки слабо подходят к цифрам, а 1 канал растянут в RGB и 224×224.")

plt.figure(); plt.plot(range(1, len(loss_c) + 1), loss_c, marker="o", label="CNN")
plt.plot(range(1, len(loss_a) + 1), loss_a, marker="s", label="AlexNet")
plt.xlabel("Эпоха"); plt.ylabel("CrossEntropyLoss"); plt.title("Ошибка на обучении"); plt.legend(); plt.grid(True)

raw = datasets.MNIST(DATA, False)
pil, y = (to_digit(Image.open(sys.argv[1])), None) if len(sys.argv) > 1 else raw[random.randrange(len(raw))]
with torch.no_grad():
    p_c = cnn(tf_c(pil).unsqueeze(0).to(device)).argmax(1).item()
    p_a = alex(tf_a(pil).unsqueeze(0).to(device)).argmax(1).item()
plt.figure(); plt.imshow(pil, cmap="gray"); plt.axis("off")
plt.title(f"CNN={p_c}, AlexNet={p_a}" + ("" if y is None else f", истина={y}"))
plt.show()
