import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import SimpleCNN

EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODEL_PATH = 'stl10_cnn.pth'
PLOT_PATH = 'training_metrics.png'
SEED = 42

torch.manual_seed(SEED)
device = torch.device('cpu')

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print('Загрузка датасета STL-10...')
torchvision.datasets.STL10.url = 'https://huggingface.co/datasets/xingslong/stl10_binary/resolve/main/stl10_binary.tar.gz'
train_dataset = torchvision.datasets.STL10(
    root='./data', split='train', download=True, transform=train_transform
)
test_dataset = torchvision.datasets.STL10(
    root='./data', split='test', download=True, transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses, test_losses = [], []
train_accs, test_accs = [], []


def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


print(f'Старт обучения на {device} ({EPOCHS} эпох)...')
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = 100.0 * correct / total
    test_loss, test_acc = evaluate(test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)

    print(
        f'Эпоха [{epoch:2d}/{EPOCHS}] | '
        f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | '
        f'Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%'
    )

torch.save(model.state_dict(), MODEL_PATH)
print(f'\nМодель сохранена в {MODEL_PATH}')
print(f'Итоговая точность на тестовой выборке: {test_accs[-1]:.2f}%')

epochs_range = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs_range, train_losses, 'o-', label='Train Loss')
ax1.plot(epochs_range, test_losses, 's-', label='Test Loss')
ax1.set_title('Изменение ошибки (CrossEntropyLoss)')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(epochs_range, train_accs, 'o-', label='Train Accuracy')
ax2.plot(epochs_range, test_accs, 's-', label='Test Accuracy')
ax2.set_title('Качество модели (Accuracy)')
ax2.set_xlabel('Эпоха')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
print(f'График сохранён в {PLOT_PATH}')
plt.show()