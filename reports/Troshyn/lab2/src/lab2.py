import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
import matplotlib.pyplot as plt
from PIL import Image

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1.0
RHO = 0.9

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225
)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)


test_dataset = torchvision.datasets.CIFAR10(
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
print()

weights = DenseNet121_Weights.DEFAULT
model = densenet121(
    weights=weights
)

for parameter in model.parameters():
    parameter.requires_grad = False

for parameter in model.features.denseblock4.parameters():
    parameter.requires_grad = True

for parameter in model.features.norm5.parameters():
    parameter.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)

model = model.to(DEVICE)

print()
#print(model)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adadelta(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.1,
    rho=0.9,
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

        running_loss += (loss.item() * images.size(0))
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = (running_loss / total)
    train_accuracy = (100.0 * correct / total)

    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Предсказание
            outputs = model(images)

            # Ошибка
            loss = criterion(outputs, labels)
            test_loss_sum += (loss.item() * images.size(0))

            # Предсказанные классы
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_loss = (test_loss_sum / test_total)

    test_accuracy = (100.0 * test_correct / test_total)

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
        torch.save(model.state_dict(), "best_densenet121_cifar10.pth")

print()
print(f"Best test accuracy: {best_accuracy:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    label="Train Loss"
)
plt.plot(
    range(1, EPOCHS + 1),
    test_losses,
    label="Test Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("изменение функции ошибки")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("loss.png", dpi=300)
plt.show()

model.load_state_dict(
    torch.load(
        "best_densenet121_cifar10.pth",
        map_location=DEVICE
    )
)
model.eval()

def denormalize(image):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image.cpu()
    image = (image * std + mean)
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

plt.savefig("predictions.png", dpi=300)
plt.show()