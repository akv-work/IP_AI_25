import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import densenet121, DenseNet121_Weights
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = "best_densenet121_cifar10.pth"

IMAGE_PATH = "cat.jpg"

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])

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

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully")

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image:", IMAGE_PATH)
print("Original size:", image.size)

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(DEVICE)

with torch.no_grad():
    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

class_name = classes[predicted_class]

probability = probabilities[
    0,
    predicted_class
].item() * 100


print()
print("Prediction:", class_name)
print(f"Probability: {probability:.2f}%")

plt.figure(figsize=(5, 5))
plt.imshow(
    image
)
plt.title(
    f"Pred: {classes[predicted_class]}\n"
    f"{probability:.2f}%"
)
plt.axis("off")
plt.tight_layout()
plt.show()