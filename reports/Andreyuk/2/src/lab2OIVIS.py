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


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
aug = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                          transforms.Lambda(lambda t: t.repeat(3, 1, 1)),
                          transforms.Normalize(MEAN, STD)])
small = transforms.Compose([transforms.Resize(28), transforms.ToTensor()])


def get_loaders(tf, batch):
    tr = datasets.MNIST(DATA, train=True,  download=True, transform=tf)
    te = datasets.MNIST(DATA, train=False, download=True, transform=tf)
    return DataLoader(tr, batch, True), DataLoader(te, batch)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10))
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

def normalize_digit(img):
    t = transforms.ToTensor()(img.convert("L"))
    return transforms.ToPILImage()(1 - t if t.mean() > 0.5 else t)


cnn = SmallCNN().to(dev)
tr_c, te_c = get_loaders(small, 64)
hist_cnn = train(cnn, tr_c, optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9), 3, "CNN")
acc_cnn = accuracy(cnn, te_c)


alex = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
for p in alex.features.parameters(): p.requires_grad = False
alex.classifier[6] = nn.Linear(4096, 10)
alex = alex.to(dev)
tr_a, te_a = get_loaders(aug, 32)
hist_alex = train(alex, tr_a, optim.SGD(alex.classifier.parameters(), lr=0.01, momentum=0.9), 2, "AlexNet")
acc_alex = accuracy(alex, te_a)

print(f"\nCNN (ЛР1): {acc_cnn:.2f}%   AlexNet (ЛР2): {acc_alex:.2f}%")

plt.figure()
plt.plot(range(1, len(hist_cnn)+1),  hist_cnn,  marker="o", label="CNN")
plt.plot(range(1, len(hist_alex)+1), hist_alex, marker="s", label="AlexNet")
plt.xlabel("Эпоха"); plt.ylabel("CrossEntropyLoss")
plt.title("Ошибка на обучении"); plt.legend(); plt.grid(True)

raw = datasets.MNIST(DATA, False)
if len(sys.argv) > 1:
    img, truth = normalize_digit(Image.open(sys.argv[1])), None
else:
    img, truth = raw[random.randrange(len(raw))]

with torch.no_grad():
    pred_cnn  = cnn(small(img).unsqueeze(0).to(dev)).argmax(1).item()
    pred_alex = alex(aug(img).unsqueeze(0).to(dev)).argmax(1).item()

plt.figure(); plt.imshow(img, cmap="gray"); plt.axis("off")
plt.title(f"CNN={pred_cnn}, AlexNet={pred_alex}"
          + ("" if truth is None else f", истина={truth}"))
plt.show()