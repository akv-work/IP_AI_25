import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

BATCH_SIZE = 128
EPOCHS = 20
LR = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = (0.5071, 0.4865, 0.4409)
std = (0.2673, 0.2564, 0.2762)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_data = datasets.CIFAR100("./data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR100("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

classes = train_data.classes

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 100)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

train_losses = []  
test_losses = []   
accuracies = []

print("Обучение:")

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
print(f"Итоговая точность: {accuracies[-1]:.2f}%")

plt.plot(range(1, EPOCHS + 1), train_losses)  
plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Изменение ошибки при обучении")
plt.grid()
plt.show()

model.eval()
plt.figure(figsize=(10, 6))
for i in range(6):
    image, label = test_data[
        torch.randint(len(test_data), (1,)).item()
    ]

    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))
        prediction = output.argmax(dim=1).item()

    image_show = image * torch.tensor(std).view(3, 1, 1)
    image_show += torch.tensor(mean).view(3, 1, 1)
    image_show = image_show.clamp(0, 1)

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
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
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

predict_image("images.jpg")