# Time    : 2022.07.18 下午 03:55
# Author  : Vandaci(cnfendaki@qq.com)
# File    : for_voc.py
# Project : ClassicNetwork
import os
import csv
import numpy as np

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


if __name__ == '__main__':
    write_train_label(ANNOTATION_PATH, IMAGES_PATH)
    pass
