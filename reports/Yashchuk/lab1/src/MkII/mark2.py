
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk, filedialog

# Пути к файлам рядом со скриптом
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

# Выбор устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

# Подготовка изображений
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Названия классов
classes = (
    "Самолет", "Автомобиль", "Птица", "Кот", "Олень",
    "Собака", "Лягушка", "Лошадь", "Корабль", "Грузовик"
)

# Архитектура нейросети
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Создание модели
model = CNN().to(device)


# Обучение модели
def train_model():
    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True,
        transform=transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=64, shuffle=False
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    epochs = 10
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Проверка на тестовой выборке
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print(
            f"Эпоха {epoch + 1}/{epochs}, "
            f"Ошибка: {epoch_loss:.4f}, "
            f"Точность: {accuracy:.2f}%"
        )

    # Сохранение обученных весов
    torch.save(model.state_dict(), MODEL_PATH)
    print("Модель сохранена:", MODEL_PATH)

    # График ошибки
    plt.plot(range(1, epochs + 1), train_losses, marker="o")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.title("Изменение ошибки при обучении")
    plt.grid()
    plt.savefig(os.path.join(BASE_DIR, "loss.png"))
    plt.show()


# Загрузка готовых весов
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Файл model.pth не найден.")
        print("Сначала запусти обучение.")
        return False

    state = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    print("Готовая модель загружена.")
    return True


# Выбор и классификация фотографии
def predict_image():
    root = Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Выбери изображение",
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png *.bmp"),
            ("Все файлы", "*.*")
        ]
    )

    root.destroy()

    if not path:
        print("Файл не выбран.")
        return

    image = Image.open(path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    result = classes[predicted.item()]
    confidence = confidence.item() * 100

    print("Файл:", path)
    print("Результат:", result)
    print(f"Уверенность: {confidence:.2f}%")

    plt.imshow(image)
    plt.title(f"{result}: {confidence:.2f}%")
    plt.axis("off")
    plt.show()


# Главное меню
print("\n1. Обучить модель")
print("2. Проверить изображение")
choice = input("Выбери режим: ")

if choice == "1":
    train_model()

elif choice == "2":
    if load_model():
        predict_image()

else:
    print("Неизвестный режим.")