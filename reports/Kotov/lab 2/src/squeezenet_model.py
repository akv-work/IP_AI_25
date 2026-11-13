import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights


def build_squeezenet(num_classes=10, pretrained=True, freeze_features=True):
    weights = SqueezeNet1_1_Weights.DEFAULT if pretrained else None
    model = squeezenet1_1(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = num_classes
    return model