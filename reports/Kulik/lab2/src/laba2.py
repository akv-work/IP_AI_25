import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34, ResNet34_Weights
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
PATIENCE = 3
NUM_CLASSES = 10

MODEL_PATH = "best_resnet34_fashion.pth"

classes = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def main():
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", DEVICE)

    # ---------- ПРЕДОБРАБОТКА ----------
    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(128, padding=8, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform_test
    )

    num_workers = 2 if DEVICE.type == "cuda" else 0
    pin_memory = DEVICE.type == "cuda"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print("Train images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    weights = ResNet34_Weights.IMAGENET1K_V1
    model = resnet34(weights=weights)

    in_features = model.fc.in_features  # 512
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    best_accuracy = 0.0
    best_epoch = 0
    no_improve = 0

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=pin_memory)
            labels = labels.to(DEVICE, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE, non_blocking=pin_memory)
                labels = labels.to(DEVICE, non_blocking=pin_memory)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

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

        scheduler.step()

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Сохранена лучшая модель (Acc: {best_accuracy:.2f}%)")
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

    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, test_losses, label="Test Loss")
    plt.axvline(best_epoch, color="green", linestyle="--",
                label=f"Лучшая эпоха ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Изменение функции ошибки (ResNet34, Fashion-MNIST)")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("loss.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
    plt.plot(epochs_range, test_accuracies, label="Test Accuracy")
    plt.axvline(best_epoch, color="green", linestyle="--",
                label=f"Лучшая эпоха ({best_epoch})")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)")
    plt.title("Изменение точности классификации (ResNet34, Fashion-MNIST)")
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig("accuracy.png", dpi=300)
    plt.show()

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    def denormalize(image):
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        image = image.cpu() * std + mean
        return torch.clamp(image, 0, 1)

    images, labels = next(iter(test_loader))
    images_gpu = images.to(DEVICE)

    with torch.no_grad():
        outputs = model(images_gpu)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)

    plt.figure(figsize=(14, 8))
    for i in range(12):
        image = denormalize(images[i]).permute(1, 2, 0).numpy()

        plt.subplot(3, 4, i + 1)
        plt.imshow(image, cmap="gray")
        predicted_class = classes[predictions[i].item()]
        real_class = classes[labels[i].item()]
        prob = probabilities[i, predictions[i]].item() * 100
        color = "green" if predicted_class == real_class else "red"
        plt.title(
            f"Pred: {predicted_class}\nReal: {real_class}\n{prob:.2f}%",
            color=color, fontsize=9
        )
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=300)
    plt.show()

    def predict_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

        image = Image.open(image_path).convert("RGB")
        original_image = image.copy()
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)

        predicted_class = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0, predicted_class].item()

        plt.figure(figsize=(5, 5))
        plt.imshow(original_image)
        plt.title(
            f"Предсказание: {classes[predicted_class]}\n"
            f"Вероятность: {probability * 100:.3f}%"
        )
        plt.axis("off"); plt.tight_layout()
        plt.savefig("custom_prediction.png", dpi=300)
        plt.show()

        print("\nПредсказанный класс:", classes[predicted_class])
        print("Вероятность:", f"{probability * 100:.3f}%")

    raw_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=False,
        transform=transforms.ToTensor()
    )
    for i in range(len(raw_dataset)):
        img, label = raw_dataset[i]
        if label == 7:
            torchvision.utils.save_image(img, "shoe.jpg")
            print(f"Сохранено: shoe.jpg (индекс {i}, класс Sneaker)")
            break

    predict_image("shoe.jpg")

if __name__ == '__main__':
    main()