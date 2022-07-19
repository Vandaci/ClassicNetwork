# Time    : 2022.07.19 下午 03:07
# Author  : Vandaci(cnfendaki@qq.com)
# File    : test.py
# Project : ClassicNetwork
import torch
from train import ResNetforVOC
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from torch.nn import functional as F

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    net = ResNetforVOC()
    net.load_state_dict(torch.load('../data/resnet_voc.pth', map_location=DEVICE))
    net.eval()
    image = read_image('../images/000008.jpg', mode=ImageReadMode.RGB) / 255.
    transform = transforms.Resize((250, 250))
    image = transform(image)
    image = torch.unsqueeze(image, dim=0)
    with torch.no_grad():
        y = net(image)
        y = F.sigmoid(y)
        pass
