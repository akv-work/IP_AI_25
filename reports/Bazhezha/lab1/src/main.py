import random, ssl, sys, certifi, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
tfm = transforms.ToTensor()
train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tfm)
train_ld, test_ld = DataLoader(train_ds, 64, True), DataLoader(test_ds, 256)

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

def to_mnist(path):
    img = transforms.Compose([transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor()])(Image.open(path))
    return 1 - img if img.mean() > 0.5 else img

model, crit = CNN().to(device), nn.CrossEntropyLoss()
opt, losses = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9), []
for epoch in range(5):
    model.train(); s = 0
    for x, y in train_ld:
        x, y = x.to(device), y.to(device)
        loss = crit(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        s += loss.item()
    losses.append(s / len(train_ld))
    print(f"epoch {epoch + 1} loss {losses[-1]:.4f}")

model.eval(); ok = n = 0
with torch.no_grad():
    for x, y in test_ld:
        pred = model(x.to(device)).argmax(1).cpu()
        ok += (pred == y).sum().item(); n += len(y)
acc = 100 * ok / n
print(f"test accuracy: {acc:.2f}%")
print("SOTA MNIST ≈ 99.8% (CLoVE, 2025; ансамбли простых CNN ~99.7–99.8%).")
print(f"Вывод: простая СНС ({acc:.2f}%) близка к классическим CNN ~98–99%, но ниже SOTA из-за малой глубины, 5 эпох и отсутствия аугментаций.")

plt.figure(); plt.plot(range(1, len(losses) + 1), losses, marker="o")
plt.xlabel("Эпоха"); plt.ylabel("CrossEntropyLoss"); plt.title("Ошибка на обучении"); plt.grid(True)

img, label = (to_mnist(sys.argv[1]), None) if len(sys.argv) > 1 else test_ds[random.randrange(len(test_ds))]
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
plt.figure(); plt.imshow(img.squeeze(), cmap="gray"); plt.axis("off")
plt.title(f"предсказание={pred}" + ("" if label is None else f", истина={label}"))
plt.show()
