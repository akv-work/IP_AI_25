import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# -----------------------------
# Настройки
# -----------------------------
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 2e-3
PATIENCE = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "best.pth"

print("Device:", DEVICE)

classes = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
)

# -----------------------------
# Аугментация
# -----------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])

# -----------------------------
# Датасеты
# -----------------------------
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))

# -----------------------------
# Модель
# -----------------------------
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32 * 7 * 7,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                10
            )
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


model = CNN().to(DEVICE)
print(model)

# -----------------------------
# Функция потерь и оптимизатор
# -----------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# -----------------------------
# Обучение
# -----------------------------
best_accuracy = 0.0
best_epoch = 0
no_improve = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100.0 * correct / total

    # проверка на тестовой выборке

    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            test_loss_sum += (
                loss.item() *
                images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            test_total += labels.size(0)

            test_correct += (
                predicted == labels
            ).sum().item()

    test_loss = test_loss_sum / test_total
    test_accuracy = 100.0 * test_correct / test_total

    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Test Loss: {test_loss:.4f} "
        f"Test Acc: {test_accuracy:.2f}%"
    )

    # сохранение лучшей модели + early stopping

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy
        best_epoch = epoch + 1
        no_improve = 0

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

    else:

        no_improve += 1
        print(f"  Нет улучшения ({no_improve}/{PATIENCE})")

        if no_improve >= PATIENCE:

            print(
                f"\nEarly stopping на эпохе {epoch + 1}. "
                f"Лучшая эпоха: {best_epoch} "
                f"(Test Acc: {best_accuracy:.2f}%)"
            )
            break

print(f"\nBest test accuracy: {best_accuracy:.2f}%")

# -----------------------------
# Графики
# -----------------------------
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    train_losses,
    label="Train Loss"
)

plt.plot(
    epochs_range,
    test_losses,
    label="Test Loss"
)

plt.axvline(
    best_epoch,
    color="green",
    linestyle="--",
    label=f"Лучшая эпоха ({best_epoch})"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Изменение функции ошибки")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("loss.png", dpi=300)
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    epochs_range,
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    epochs_range,
    test_accuracies,
    label="Test Accuracy"
)

plt.axvline(
    best_epoch,
    color="green",
    linestyle="--",
    label=f"Лучшая эпоха ({best_epoch})"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Изменение точности классификации")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("accuracy.png", dpi=300)
plt.show()

# -----------------------------
# Загрузка лучшей модели
# -----------------------------
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

# -----------------------------
# Визуализация случайных изображений из тестовой выборки
# -----------------------------
def denormalize(image):

    mean = torch.tensor([0.5]).view(1, 1, 1)
    std = torch.tensor([0.5]).view(1, 1, 1)

    image = image.cpu()

    image = image * std + mean

    return torch.clamp(image, 0, 1)


images, labels = next(iter(test_loader))

images_gpu = images.to(DEVICE)

with torch.no_grad():

    outputs = model(images_gpu)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    _, predictions = torch.max(
        outputs,
        1
    )

plt.figure(figsize=(12, 8))

for i in range(12):

    image = denormalize(images[i])

    image = image.squeeze().numpy()

    plt.subplot(3, 4, i + 1)

    plt.imshow(image, cmap="gray")

    predicted_class = classes[predictions[i].item()]
    real_class = classes[labels[i].item()]

    probability = probabilities[
        i,
        predictions[i]
    ].item() * 100

    plt.title(
        f"Pred: {predicted_class}\n"
        f"Real: {real_class}\n"
        f"{probability:.3f}%"
    )

    plt.axis("off")

plt.tight_layout()
plt.savefig("predictions.png", dpi=300)
plt.show()

# -----------------------------
# Классификация произвольного изображения
# -----------------------------
def predict_image(image_path):

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,),
            (0.5,)
        )
    ])

    image = Image.open(
        image_path
    ).convert("RGB")

    original_image = image.copy()

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(DEVICE)

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    probability = probabilities[
        0,
        predicted_class
    ].item()

    plt.figure(figsize=(5, 5))

    plt.imshow(original_image)

    plt.title(
        f"Предсказание: "
        f"{classes[predicted_class]}\n"
        f"Вероятность: "
        f"{probability * 100:.3f}%"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print()
    print(
        "Предсказанный класс:",
        classes[predicted_class]
    )

    print(
        "Вероятность:",
        f"{probability * 100:.3f}%"
    )


# Датасет без transform — «сырые» картинки
raw_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=False,
    transform=transforms.ToTensor()   # только to tensor, без normalize
)

for i in range(len(raw_dataset)):
    img, label = raw_dataset[i]
    if label == 7:
        torchvision.utils.save_image(img, "shoe.jpg")
        print(f"Сохранено: shoe.jpg (индекс {i}, класс Sneaker)")
        break

predict_image("shoe.jpg")