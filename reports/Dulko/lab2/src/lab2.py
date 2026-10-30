import sys
from pathlib import Path

from PIL import Image
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "lab1" / "src"))
from lab1 import CNN, classify, restore, seed_everything, train
import matplotlib.pyplot as plt


def densenet():
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    model.requires_grad_(False)
    model.classifier = nn.Linear(model.classifier.in_features, 10)
    return model


def compare(path):
    image = Image.open(path).convert("RGB")
    models = [
        (restore(CNN(), ROOT.parent / "lab1" / "results" / "model.pth"), "CNN", False),
        (restore(densenet(), ROOT / "results" / "model.pth", True), "DenseNet121", True),
    ]
    titles = []
    for model, name, transfer in models:
        label, score = classify(model, image, transfer)
        titles.append(f"{name}: {label} ({score:.1%})")
        print(f"{name}: {label} ({score:.1%})")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.imshow(image)
    ax.set_title("\n".join(titles))
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    seed_everything()
    if len(sys.argv) > 1:
        compare(sys.argv[1])
    else:
        train(densenet(), ROOT / "results", transfer=True, epochs=8)
