# Time    : 2022.07.18 上午 10:22
# Author  : Vandaci(cnfendaki@qq.com)
# File    : resnet.py
# Project : ClassicNetwork
import torch
from torchvision.models import resnet18
from torchsummary import summary
from torch.utils.tensorboard import writer

if __name__ == '__main__':
    net = resnet18(weights=None)
    # summary(net, (3, 224, 224), device='cpu')
    # wter = writer.SummaryWriter('./runs/resnet18', comment="ResNet18")
    inputs = torch.randn((1, 3, 224, 224))
    # wter.add_graph(net, inputs)
    # wter.close()
    pass
