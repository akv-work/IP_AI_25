import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt
import os

batchsize = 64
epochs = 5
Lr = 0.001
model_file = "resnet18_mnist.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False)

print("Обучающая выборка:", len(train_data))
print("Тестовая выборка:", len(test_data))

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

losses = []
accuracies = []

print("\nВыберите режим:")
print("1 - Обучить модель заново")
print("2 - Загрузить сохранённую модель")

choice = input("Ваш выбор: ")

if choice == "1":
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=Lr)

    print("\nОбучение:")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch + 1) % 100 == 0:
                print(
                    f"Эпоха {epoch + 1}/{epochs} | "
                    f"Батч {batch + 1}/{len(train_loader)} | "
                    f"Ошибка: {loss.item():.4f}"
                )

        loss_value = total_loss / len(train_loader)
        losses.append(loss_value)

        model.eval()
        correct = total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                prediction = output.argmax(dim=1)

                correct += (prediction == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total * 100
        accuracies.append(accuracy)

        print(
            f"Эпоха {epoch + 1}/{epochs} завершена | "
            f"Ошибка: {loss_value:.4f} | "
            f"Точность: {accuracy:.2f}%"
        )

    print(f"\nИтоговая точность: {accuracies[-1]:.2f}%")

    torch.save({
        "model_state": model.state_dict(),
        "losses": losses,
        "accuracies": accuracies
    }, model_file)

    print(f"Модель сохранена: {model_file}")

elif choice == "2":
    if not os.path.exists(model_file):
        print(f"\nФайл {model_file} не найден.")
        print("Сначала обучите модель, выбрав режим 1.")
        exit()

    checkpoint = torch.load(model_file, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    losses = checkpoint["losses"]
    accuracies = checkpoint["accuracies"]

    print(f"\nМодель загружена: {model_file}")
    print(f"Сохранённая точность: {accuracies[-1]:.2f}%")

else:
    print("\nНеверный выбор.")
    exit()

if losses:
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.title("Изменение ошибки при обучении ResNet18")
    plt.grid()
    plt.show()

model.eval()
plt.figure(figsize=(9, 6))

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

for i in range(6):
    image, label = test_data[torch.randint(len(test_data), (1,)).item()]

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        prediction = output.argmax(dim=1).item()

    image_show = (image.cpu() * std + mean).clamp(0, 1)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image_show.permute(1, 2, 0))
    plt.title(f"Правильная: {label}, сеть: {prediction}")
    plt.axis("off")

plt.tight_layout()
plt.show()

folder = "my_digits"

if os.path.exists(folder):
    print("\nПроверка изображений из my_digits:")

    for file in os.listdir(folder):
        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        path = os.path.join(folder, file)
        image = Image.open(path).convert("L")

        if image.getpixel((0, 0)) > 128:
            image = ImageOps.invert(image)

        image = image.filter(ImageFilter.MinFilter(3))

        box = image.getbbox()
        if box:
            image = image.crop(box)

        w, h = image.size
        scale = 20 / max(w, h)
        image = image.resize(
            (max(1, int(w * scale)), max(1, int(h * scale)))
        )

        mnist_image = Image.new("L", (28, 28), 0)

        x = (28 - image.width) // 2
        y = (28 - image.height) // 2

        mnist_image.paste(image, (x, y))

        image = mnist_image.resize((64, 64))

        image_tensor = transforms.ToTensor()(image)
        image_tensor = image_tensor.repeat(3, 1, 1)

        image_tensor = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )(image_tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = output.argmax(dim=1).item()

        print(f"{file} - {prediction}")

        plt.figure(figsize=(3, 3))
        plt.imshow(mnist_image, cmap="gray")
        plt.title(f"{file}: сеть думает это {prediction}")
        plt.axis("off")
        plt.show()

else:
    print("\nПапка my_digits не найдена")