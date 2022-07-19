# Time    : 2022.07.18 下午 03:55
# Author  : Vandaci(cnfendaki@qq.com)
# File    : for_voc.py
# Project : ClassicNetwork
import os
import csv
import numpy as np
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
import torch

CLASS_NAME = ['person',
              'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
              'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
              'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']
ANNOTATION_PATH = r'D:\Datasets\PascalVOC\2007\VOCdevkit\VOC2007\ImageSets\Main'
IMAGES_PATH = r'D:\Datasets\PascalVOC\2007\VOCdevkit\VOC2007\JPEGImages'


def write_train_label(annotation_path, images_path):
    images_list = os.listdir(images_path)
    label = np.empty((len(images_list), 21), dtype=object)
    # 将文件名写入到numpy数组
    for i, img_name in enumerate(images_list):
        label[i, 0] = img_name
    for i, cls in enumerate(CLASS_NAME, start=1):
        with open(os.path.join(annotation_path, cls + "_trainval.txt"), 'r') as f:
            label_lines = f.readlines()
            for line in label_lines:
                labeltxt = line.strip('\n').split(' ')
                im_name = labeltxt[0] + ".jpg"
                im_label = labeltxt[-1]
                if im_label == '1':
                    label[label[:, 0] == im_name, i] = '1'
                else:
                    label[label[:, 0] == im_name, i] = '0'
    with open('label.csv', 'w+', newline='') as csfile:
        csvwriter = csv.writer(csfile)
        csvwriter.writerows(label)


class VOCClassifyDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None, target_transform=None):
        self.img_dir = img_dir
        # 跳过表头
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[item, 0])
        image = read_image(img_path) / 255.0
        label = self.img_labels.iloc[item].iloc[1:]
        label = np.array(label, dtype=float)
        label = torch.tensor(label)
        # label = label.iloc[1:]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.img_labels)


if __name__ == '__main__':
    # write_train_label(ANNOTATION_PATH, IMAGES_PATH)

    # transform = torchvision.transforms.ToTensor()
    vocdataset = VOCClassifyDataset(IMAGES_PATH, 'label.csv',)
    img, label = vocdataset[0]
    pass
