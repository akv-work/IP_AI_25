import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

MODEL_PATH = "best_cifar100.pth"

print("Device:", DEVICE)


CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD  = (0.2673, 0.2564, 0.2762)

_dataset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True
)
classes = _dataset.classes
NUM_CLASSES = len(classes)



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


model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()

print("Model loaded:", MODEL_PATH)


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
])



def predict_image(image_path):
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

    print()
    print("Результат классификации")
    print("Изображение:", image_path)
    print("Предсказанный класс:", classes[predicted_class])
    print("Вероятность:", f"{probability * 100:.2f}%")

    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.title(
        f"Pred: {classes[predicted_class]}\n"
        f"{probability * 100:.2f}%"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()



predict_image("sea.jpg")