import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1.0
RHO = 0.9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])


train_dataset = torchvision.datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)


classes = train_dataset.classes
NUM_CLASSES = len(classes)


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
print("Number of classes:", NUM_CLASSES)



class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = CNN(num_classes=NUM_CLASSES).to(DEVICE)
print(model)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    rho=RHO,
    weight_decay=1e-4
)

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

best_accuracy = 0.0


for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

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
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

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

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_cifar100.pth")

print(f"\nBest test accuracy: {best_accuracy:.2f}%")


plt.figure(figsize=(10, 5))

plt.plot(range(1, EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, EPOCHS + 1), test_losses, label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Изменение функции ошибки (CIFAR-100)")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("loss_cifar100.png", dpi=300)
plt.show()


plt.figure(figsize=(10, 5))

plt.plot(range(1, EPOCHS + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, EPOCHS + 1), test_accuracies, label="Test Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Изменение точности классификации (CIFAR-100)")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("accuracy_cifar100.png", dpi=300)
plt.show()


model.load_state_dict(
    torch.load("best_cifar100.pth", map_location=DEVICE)
)
model.eval()



def denormalize(image):
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)

    image = image.cpu()
    image = image * std + mean

    return torch.clamp(image, 0, 1)



images, labels = next(iter(test_loader))
images_gpu = images.to(DEVICE)

with torch.no_grad():
    outputs = model(images_gpu)
    probabilities = torch.softmax(outputs, dim=1)
    _, predictions = torch.max(outputs, 1)

plt.figure(figsize=(12, 8))

for i in range(12):
    image = denormalize(images[i])
    image = image.permute(1, 2, 0).numpy()

    plt.subplot(3, 4, i + 1)
    plt.imshow(image)

    predicted_class = classes[predictions[i].item()]
    real_class = classes[labels[i].item()]
    probability = probabilities[i, predictions[i]].item() * 100

    plt.title(
        f"Pred: {predicted_class}\n"
        f"Real: {real_class}\n"
        f"{probability:.1f}%"
    )
    plt.axis("off")

plt.tight_layout()
plt.savefig("predictions_cifar100.png", dpi=300)
plt.show()



def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    ])

    image = Image.open(image_path).convert("RGB")
    original_image = image.copy()

    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(probabilities, dim=1).item()
    probability = probabilities[0, predicted_class].item()

    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.title(
        f"Предсказание: {classes[predicted_class]}\n"
        f"Вероятность: {probability * 100:.2f}%"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print()
    print("Предсказанный класс:", classes[predicted_class])
    print("Вероятность:", f"{probability * 100:.2f}%")


predict_image("sea.jpg")