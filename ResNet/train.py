# Time    : 2022.07.19 上午 08:24
# Author  : Vandaci(cnfendaki@qq.com)
# File    : train.py
# Project : ClassicNetwork
from torchvision.models import resnet18
import torch.nn as nn
from data.for_voc import VOCClassifyDataset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms


class ResNetforVOC(nn.Module):
    def __init__(self):
        super(ResNetforVOC, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, 20)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    net = ResNetforVOC()
    net.train()
    # from torchsummary.torchsummary import summary
    #
    # summary(net, (3, 250, 250))
    img_dir = r'D:\Datasets\PascalVOC\2007\VOCdevkit\VOC2007\JPEGImages'
    transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop((250, 250))
    ])
    train_set = VOCClassifyDataset(img_dir, '../data/label.csv', transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)  # 在此之前要确保图片大小尺寸批量化一致
    epoch = range(1, 11)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for e in epoch:
        print(f'Epoch {e}:')
        for i, train_data in enumerate(train_loader):
            y = train_data[1]
            x = train_data[0]
            # forward
            y_pred = net(x)
            loss = loss_fn(y_pred, y)
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 4 == 0:
                print(f'loss {loss:.4f}')
    pass
