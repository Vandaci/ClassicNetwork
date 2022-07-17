import torch
from torch import nn


# 输入为3×224×224

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        return self.sequential(x)


if __name__ == "__main__":
    from torchvision import models

    net = models.alexnet()
    model = AlexNet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    pass
