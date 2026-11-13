import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# 1. НАСТРОЙКИ ПРОГРАММЫ

DATA_DIR = "./data"
RESULTS_DIR = "./results"
MODEL_PATH = "best_densenet121_stl10.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

CLASS_NAMES = (
    "самолёт",
    "птица",
    "автомобиль",
    "кошка",
    "олень",
    "собака",
    "лошадь",
    "обезьяна",
    "корабль",
    "грузовик",
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

os.makedirs(RESULTS_DIR, exist_ok=True)

# Преобразования изображений для обучения и проверки.
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# 2. СОЗДАНИЕ МОДЕЛИ

def create_model():
    """Создаёт DenseNet121 с предобученными весами ImageNet."""
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    # Замораживаем параметры исходной сети.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Размораживаем последний плотный блок и финальную нормализацию.
    for parameter in model.features.denseblock4.parameters():
        parameter.requires_grad = True

    for parameter in model.features.norm5.parameters():
        parameter.requires_grad = True

    # Заменяем классификатор на вариант для 10 классов STL-10.
    model.classifier = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(CLASS_NAMES)),
    )

    return model.to(DEVICE)

# 3. ЗАГРУЗКА ДАННЫХ STL-10

def make_loaders():
    """Загружает обучающую и тестовую выборки STL-10."""
    train_dataset = datasets.STL10(
        root=DATA_DIR,
        split="train",
        download=True,
        transform=transform_train,
    )

    test_dataset = datasets.STL10(
        root=DATA_DIR,
        split="test",
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Устройство: {DEVICE}")
    print(f"Изображений для обучения: {len(train_dataset)}")
    print(f"Изображений для тестирования: {len(test_dataset)}")

    return train_loader, test_loader

# 4. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛИ

def train_one_epoch(model, loader, criterion, optimizer):
    """Выполняет один проход обучения."""
    model.train()

    # Не обновляем статистики BatchNorm в замороженных ранних блоках.
    model.features.denseblock1.eval()
    model.features.transition1.eval()
    model.features.denseblock2.eval()
    model.features.transition2.eval()
    model.features.denseblock3.eval()
    model.features.transition3.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        count = labels.size(0)
        total_loss += loss.item() * count
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += count

    return total_loss / total, 100.0 * total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Вычисляет ошибку и точность без обновления весов."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        count = labels.size(0)
        total_loss += loss.item() * count
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += count

    return total_loss / total, 100.0 * total_correct / total

# 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ

def save_graphs(history):
    """Строит и сохраняет графики ошибки и точности."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Обучение")
    axes[0].plot(epochs, history["test_loss"], marker="o", label="Тестирование")
    axes[0].set_title("Функция ошибки")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("CrossEntropyLoss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Обучение")
    axes[1].plot(epochs, history["test_acc"], marker="o", label="Тестирование")
    axes[1].set_title("Точность классификации")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Точность, %")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_graphs.png"), dpi=300)
    plt.close(fig)


def save_history(history):
    """Записывает значения метрик в CSV-файл."""
    path = os.path.join(RESULTS_DIR, "training_history.csv")

    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "Эпоха",
            "Ошибка обучения",
            "Точность обучения (%)",
            "Ошибка тестирования",
            "Точность тестирования (%)",
        ])

        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.2f}",
                f"{history['test_loss'][i]:.6f}",
                f"{history['test_acc'][i]:.2f}",
            ])


def save_test_visualization(model, test_loader):
    """Сохраняет примеры предсказаний для 12 тестовых изображений."""
    model.eval()
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images.to(DEVICE))
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)

    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    for i, ax in enumerate(axes.flat):
        image = images[i].cpu() * std + mean
        image = torch.clamp(image, 0, 1).permute(1, 2, 0).numpy()
        ax.imshow(image)

        predicted_class = CLASS_NAMES[predictions[i].item()]
        actual_class = CLASS_NAMES[labels[i].item()]
        confidence = probabilities[i, predictions[i]].item() * 100

        ax.set_title(
            f"Предсказание: {predicted_class}\n"
            f"Правильный класс: {actual_class}\n"
            f"{confidence:.1f}%"
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "test_predictions.png"), dpi=300)
    plt.close(fig)

# 6. ОБУЧЕНИЕ МОДЕЛИ

def train_model():
    """Обучает DenseNet121 и сохраняет лучшую версию модели."""
    print("\n[1/4] Загрузка набора данных STL-10...")
    train_loader, test_loader = make_loaders()

    print("\n[2/4] Загрузка предобученной DenseNet121...")
    model = create_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4,
    )

    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }
    best_accuracy = -1.0

    print("\n[3/4] Начало обучения...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(
            f"Эпоха [{epoch + 1}/{EPOCHS}] | "
            f"Ошибка обучения: {train_loss:.4f} | "
            f"Точность обучения: {train_acc:.2f}% | "
            f"Ошибка теста: {test_loss:.4f} | "
            f"Точность теста: {test_acc:.2f}%"
        )

        # Сохраняем модель с наилучшей точностью на тестовой выборке.
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "best_test_accuracy": best_accuracy,
                },
                MODEL_PATH,
            )

    print("\n[4/4] Сохранение графиков и истории...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    save_graphs(history)
    save_history(history)
    save_test_visualization(model, test_loader)

    print(f"\nОбучение завершено. Лучшая точность: {best_accuracy:.2f}%")
    print(f"Модель: {MODEL_PATH}")
    print(f"Графики: {RESULTS_DIR}/training_graphs.png")
    print(f"История: {RESULTS_DIR}/training_history.csv")
    print(f"Примеры предсказаний: {RESULTS_DIR}/test_predictions.png")

    return model

# 7. ЗАГРУЗКА СОХРАНЁННОЙ МОДЕЛИ

def load_saved_model():
    """Загружает модель из файла, если обучение уже выполнялось."""
    if not os.path.exists(MODEL_PATH):
        return None

    model = create_model()

    try:
        checkpoint = torch.load(
            MODEL_PATH, map_location=DEVICE, weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Загружена модель: {MODEL_PATH}")
    if "best_test_accuracy" in checkpoint:
        print(
            "Лучшая точность на тестовой выборке: "
            f"{checkpoint['best_test_accuracy']:.2f}%"
        )

    return model

# 8. КЛАССИФИКАЦИЯ ПРОИЗВОЛЬНОГО ИЗОБРАЖЕНИЯ

def predict_image(model, image_path):
    """Определяет класс изображения и выводит вероятности."""
    image = Image.open(image_path).convert("RGB")
    tensor = transform_test(image).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    order = torch.argsort(probabilities, descending=True)

    print("\nРезультат классификации:")
    print(f"Файл: {image_path}")
    print(f"Предсказанный класс: {CLASS_NAMES[order[0].item()]}")
    print(f"Уверенность: {probabilities[order[0]].item() * 100:.2f}%")
    print("\nВероятности по классам:")

    for index in order:
        class_index = index.item()
        probability = probabilities[class_index].item() * 100
        print(f"  {CLASS_NAMES[class_index]:<12} — {probability:6.2f}%")

# 9. КОНСОЛЬНОЕ МЕНЮ

def main():
    print(f"Устройство: {DEVICE}")

    model = None

    while True:
        print("\nВыберите действие:")
        print("1 — Обучить модель")
        print("2 — Классифицировать изображение")
        print("0 — Выход")

        choice = input("Ваш выбор: ").strip()

        if choice == "1":
            try:
                model = train_model()
            except Exception as error:
                print(f"\nОшибка во время обучения: {error}")

        elif choice == "2":
            if model is None:
                try:
                    model = load_saved_model()
                except Exception as error:
                    print(f"Не удалось загрузить модель: {error}")
                    continue

            if model is None:
                print("Сохранённая модель не найдена. Сначала выполните обучение.")
                continue

            image_path = input("Введите путь к изображению: ").strip().strip('"')

            if not os.path.isfile(image_path):
                print("Ошибка: файл по указанному пути не найден.")
                continue

            try:
                predict_image(model, image_path)
            except Exception as error:
                print(f"Не удалось классифицировать изображение: {error}")

        elif choice == "0":
            print("Завершение программы.")
            break

        else:
            print("Неверный пункт меню. Введите 1, 2 или 0.")


# 10. ТОЧКА ВХОДА

if __name__ == "__main__":
    main()
