import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

batchsize = 64
epochs = 20
Lr = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

# ПРЕОБРАЗОВАНИЕ

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(96, padding=4), transforms.ToTensor()])
test_transform = transforms.ToTensor()

# STL-10

train_data = datasets.STL10(
    "./data",
    split="train",
    download=True,
    transform=train_transform
)

test_data = datasets.STL10(
    "./data",
    split="test",
    download=True,
    transform=test_transform
)


train_loader = DataLoader(
    train_data,
    batch_size=batchsize,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=batchsize,
    shuffle=False
)


print("Обучающая выборка:", len(train_data))
print("Тестовая выборка:", len(test_data))

# КЛАССЫ STL-10

classes = [
    "самолет",
    "птица",
    "автомобиль",
    "кошка",
    "олень",
    "собака",
    "лошадь",
    "обезьяна",
    "корабль",
    "грузовик"
]

# CNN

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(

            nn.Flatten(),

            nn.Linear(64 * 24 * 24, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        x = self.conv(x)
        x = self.fc(x)

        return x


model = CNN().to(device)

# ФУНКЦИЯ ПОТЕРЬ И SGD

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=Lr,
    momentum=0.9
)

# ОБУЧЕНИЕ

losses = []
accuracies = []

print("\nОбучение:")

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model(images)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    loss_value = total_loss / len(train_loader)

    losses.append(loss_value)

    # ПРОВЕРКА НА ТЕСТОВОЙ ВЫБОРКЕ

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)

            prediction = output.argmax(dim=1)

            correct += (prediction == labels).sum().item()

            total += labels.size(0)

    accuracy = correct / total * 100

    accuracies.append(accuracy)

    print(
        f"Эпоха {epoch + 1}/{epochs} | "
        f"Ошибка: {loss_value:.4f} | "
        f"Точность: {accuracy:.2f}%"
    )


print(
    f"\nИтоговая точность: "
    f"{accuracies[-1]:.2f}%"
)

# ГРАФИК ОШИБКИ
plt.plot(
    range(1, epochs + 1),
    losses,
    marker="o"
)

plt.xlabel("Эпоха")
plt.ylabel("Ошибка")

plt.title(
    "Изменение ошибки при обучении"
)

plt.grid()

plt.show()

# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ

model.eval()

plt.figure(figsize=(12, 7))

for i in range(6):

    image, label = test_data[
        torch.randint(
            len(test_data),
            (1,)
        ).item()
    ]

    with torch.no_grad():

        output = model(
            image.unsqueeze(0).to(device)
        )

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = probabilities.max(dim=1)

        prediction = prediction.item()
        confidence = confidence.item() * 100

    plt.subplot(2, 3, i + 1)

    # STL-10 имеет RGB-изображения,
    # поэтому cmap="gray" здесь не нужен
    plt.imshow(
        image.permute(1, 2, 0)
    )

    plt.title(
        f"Правильная: {classes[label]}\n"
        f"Сеть: {classes[prediction]}\n"
        f"Уверенность: {confidence:.2f}%"
    )

    plt.axis("off")


plt.tight_layout()
plt.show()

# РАСПОЗНАВАНИЕ СВОИХ ИЗОБРАЖЕНИЙ

folder = "my_images"


if os.path.exists(folder):

    files = os.listdir(folder)

    for file in files:

        if not file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        path = os.path.join(
            folder,
            file
        )

        image = Image.open(
            path
        ).convert("RGB")

        # STL-10 использует изображения 96x96
        image = image.resize(
            (96, 96)
        )

        image_tensor = transforms.ToTensor()(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        ).to(device)

        with torch.no_grad():

            output = model(image_tensor)

            probabilities = torch.softmax(output, dim=1)

            confidence, prediction = probabilities.max(dim=1)

            prediction = prediction.item()
            confidence = confidence.item() * 100

        print(
            f"{file} - {classes[prediction]}"
            f"({confidence:.2f}%)"
        )

        plt.figure(figsize=(4, 4))

        plt.imshow(image)

        plt.title(
            f"{file}: "
            f"сеть думает, что это "
            f"{classes[prediction]}\n"
            f"Уверенность:{confidence:.2f}%"
        )

        plt.axis("off")

        plt.show()

else:

    print(
        "\nПапка my_images не найдена"
    )