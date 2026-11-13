import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 5
LR = 0.005                    
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
IMG_SIZE = 128               
MODEL_PATH = "best_mobilenet_v3_cifar100.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Устройство:", device)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

train_data = datasets.CIFAR100("./data", train=True,  download=True, transform=transform_train)
test_data  = datasets.CIFAR100("./data", train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

classes = train_data.classes
NUM_CLASSES = len(classes)
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.features[-1].parameters():
    param.requires_grad = True

in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
)

train_losses = []
test_losses = []
accuracies = []
best_accuracy = 0.0

print("\nОбучение MobileNet v3 на CIFAR-100:")

for epoch in range(EPOCHS):
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

    train_loss_value = total_loss / len(train_loader)
    train_losses.append(train_loss_value)
    model.eval()
    correct = 0
    total = 0
    test_loss_sum = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            test_loss_sum += loss.item()
            prediction = output.argmax(dim=1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    accuracies.append(accuracy)
    test_loss_value = test_loss_sum / len(test_loader)
    test_losses.append(test_loss_value)

    print(
        f"Эпоха {epoch + 1}/{EPOCHS} | "
        f"Train loss: {train_loss_value:.4f} | "
        f"Test loss: {test_loss_value:.4f} | "
        f"Точность: {accuracy:.2f}%"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), MODEL_PATH)

print(f"Итоговая точность: {accuracies[-1]:.2f}%")
print(f"Лучшая точность: {best_accuracy:.2f}%")
print(f"Модель сохранена: {MODEL_PATH}")

plt.plot(range(1, EPOCHS + 1), train_losses, label="Train loss")
plt.plot(range(1, EPOCHS + 1), test_losses,  label="Test loss")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Изменение ошибки при обучении MobileNet v3")
plt.legend()
plt.grid()
plt.show()

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def denormalize(image):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    image = image.cpu() * std + mean
    return torch.clamp(image, 0, 1)

plt.figure(figsize=(10, 6))
for i in range(6):
    image, label = test_data[torch.randint(len(test_data), (1,)).item()]
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        prediction = output.argmax(dim=1).item()

    image_show = denormalize(image)

    plt.subplot(2, 3, i + 1)
    plt.imshow(image_show.permute(1, 2, 0))
    plt.title(
        f"Правильно: {classes[label]}\n"
        f"Сеть: {classes[prediction]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()

def predict_image(filename):
    image_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    image = Image.open(filename).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()

    plt.imshow(image)
    plt.title(f"Предсказание: {classes[prediction]}")
    plt.axis("off")
    plt.show()
    print(f"\nПредсказанный класс: {classes[prediction]}")
    
predict_image("images.jpg")