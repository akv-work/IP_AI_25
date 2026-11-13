import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, device, loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)
    return test_loss, accuracy


def plot_training(train_losses, test_losses, test_accs, epochs):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Train loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test loss')
    plt.xlabel('Эпоха')
    plt.ylabel('CrossEntropyLoss')
    plt.title('Изменение ошибки')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), test_accs, color='green', marker='o')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность, %')
    plt.title('Точность на тестовой выборке')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()


def visualize_prediction(model, dataset, device, idx=None):
    model.eval()
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    image, true_label = dataset[idx]
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Истина: {true_label} | Предсказано: {pred_label}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.bar(range(10), probs, color='steelblue')
    plt.xticks(range(10))
    plt.xlabel('Класс')
    plt.ylabel('Вероятность')
    plt.title('Распределение вероятностей')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()


def main():
    EPOCHS = 10
    BATCH_SIZE = 64
    LR = 0.01
    MOMENTUM = 0.9

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders(BATCH_SIZE)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    train_losses, test_losses, test_accs = [], [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        te_loss, te_acc = evaluate(model, device, test_loader, criterion)
        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        print(f'Epoch {epoch:2d} | train_loss={tr_loss:.4f} '
              f'| test_loss={te_loss:.4f} | test_acc={te_acc:.2f}%')

    plot_training(train_losses, test_losses, test_accs, EPOCHS)

    for _ in range(3):
        visualize_prediction(model, test_dataset, device)


if __name__ == '__main__':
    main()