
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from tkinter import Tk, filedialog


# Настройки
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DENSENET_PATH = os.path.join(BASE_DIR, "densenet121_cifar10.pth")

CLASSES = (
    "самолёт", "автомобиль", "птица", "кошка", "олень",
    "собака", "лягушка", "лошадь", "корабль", "грузовик"
)


# Предобработка изображений для DenseNet121
densenet_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Предобработка изображений для собственной CNN из лабораторной №1
cnn_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])


# Загрузка предобученной DenseNet121
def load_backbone():
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    # Замораживаем параметры предобученной сети
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    model.to(DEVICE)

    return model


# Извлечение признаков DenseNet121
def extract_features(model, loader):
    all_features = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)

            features = model.features(images)
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

            all_features.append(features.cpu())
            all_labels.append(labels)

            if (i + 1) % 100 == 0:
                print(
                    f"Обработано пакетов: {i + 1}/{len(loader)}"
                )

    return torch.cat(all_features), torch.cat(all_labels)


# Создание классификатора
def create_classifier():
    return nn.Linear(1024, 10).to(DEVICE)


# Загрузка CIFAR-10
def get_data():
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=densenet_transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=densenet_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


# Оценка классификатора
def evaluate(classifier, features, labels, criterion):
    classifier.eval()

    features = features.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():
        outputs = classifier(features)
        loss = criterion(outputs, labels)

        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()

    return loss.item(), accuracy


# Обучение DenseNet121
def train_model():
    print("\nЗагрузка CIFAR-10...")
    train_loader, test_loader = get_data()

    print("\nЗагрузка DenseNet121...")
    backbone = load_backbone()

    print("\nИзвлечение признаков обучающей выборки...")
    start_time = time.time()

    train_features, train_labels = extract_features(
        backbone, train_loader
    )

    print("\nИзвлечение признаков тестовой выборки...")
    test_features, test_labels = extract_features(
        backbone, test_loader
    )

    print(
        f"\nИзвлечение признаков завершено за "
        f"{time.time() - start_time:.1f} сек."
    )

    # Создание нового классификационного слоя
    classifier = create_classifier()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adadelta(
        classifier.parameters(),
        lr=LEARNING_RATE
    )

    train_dataset = TensorDataset(train_features, train_labels)

    train_feature_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    train_losses = []
    test_losses = []
    test_accuracies = []

    print("\nНачало обучения классификатора DenseNet121")

    for epoch in range(EPOCHS):
        classifier.train()
        total_loss = 0

        for features, labels in train_feature_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_feature_loader)

        test_loss, test_accuracy = evaluate(
            classifier,
            test_features,
            test_labels,
            criterion
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"Эпоха [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Test Loss: {test_loss:.4f} "
            f"Accuracy: {test_accuracy * 100:.2f}%"
        )

    # Сохранение обученного классификатора
    torch.save(classifier.state_dict(), DENSENET_PATH)

    print(f"\nМодель сохранена: {DENSENET_PATH}")

    # График ошибки
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Обучающая выборка")
    plt.plot(test_losses, label="Тестовая выборка")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.title("Изменение ошибки DenseNet121")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(BASE_DIR, "loss_lab2.png"))
    plt.show()

    # График точности
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, EPOCHS + 1),
        [x * 100 for x in test_accuracies],
        marker="o"
    )
    plt.xlabel("Эпоха")
    plt.ylabel("Точность, %")
    plt.title("Точность DenseNet121 на тестовой выборке")
    plt.grid()
    plt.savefig(os.path.join(BASE_DIR, "accuracy_lab2.png"))
    plt.show()

    print("\nИтоговая точность DenseNet121:",
          f"{test_accuracies[-1] * 100:.2f}%")


# Загрузка обученного классификатора
def load_classifier():
    classifier = create_classifier()

    if not os.path.exists(DENSENET_PATH):
        print("Сначала необходимо обучить DenseNet121.")
        return None

    classifier.load_state_dict(
        torch.load(DENSENET_PATH, map_location=DEVICE, weights_only=True)
    )

    classifier.eval()
    return classifier


# Предсказание DenseNet121 для одного изображения
def predict_densenet(image, backbone, classifier):
    image = densenet_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = backbone.features(image)
        features = F.relu(features, inplace=True)
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)

        outputs = classifier(features)
        prediction = torch.argmax(outputs, dim=1).item()

    return CLASSES[prediction]


# Собственная CNN из лабораторной №1
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
        return self.fc_layers(self.conv_layers(x))


def find_cnn_weights():
    paths = [
        os.path.join(BASE_DIR, "..", "..", "lab1", "src", "model.pth"),
        os.path.join(BASE_DIR, "model.pth")
    ]

    for path in paths:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path

    return None

# Проверка произвольного изображения
def test_image():
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png *.bmp")
        ]
    )

    root.destroy()

    if not file_path:
        print("Изображение не выбрано.")
        return

    image = Image.open(file_path).convert("RGB")

    # DenseNet121
    classifier = load_classifier()

    if classifier is None:
        return

    backbone = load_backbone()

    densenet_result = predict_densenet(
        image, backbone, classifier
    )

    print("\nРезультат DenseNet121:", densenet_result)

    # CNN из лабораторной №1
    cnn_path = find_cnn_weights()

    if cnn_path is None:
        print("Не найдены веса CNN из лабораторной №1.")
        return

    cnn = CNN().to(DEVICE)

    cnn.load_state_dict(
        torch.load(cnn_path, map_location=DEVICE, weights_only=True)
    )

    cnn.eval()

    input_image = cnn_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = cnn(input_image)
        prediction = torch.argmax(outputs, dim=1).item()

    print("Результат CNN из лабораторной №1:", CLASSES[prediction])


# Главное меню
def main():
    print("Устройство:", DEVICE)

    while True:
        print("\nЛабораторная работа №2")
        print("1. Обучить DenseNet121")
        print("2. Проверить произвольное изображение")
        print("0. Выход")

        choice = input("Выберите действие: ")

        if choice == "1":
            train_model()

        elif choice == "2":
            test_image()

        elif choice == "0":
            break

        else:
            print("Неверный пункт меню.")


if __name__ == "__main__":
    main()
