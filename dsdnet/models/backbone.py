import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Encoder(nn.Module):
    """ResNet18 encoder truncated before layer4."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, image):
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        return {"c1": c1, "c2": c2, "c3": c3}
