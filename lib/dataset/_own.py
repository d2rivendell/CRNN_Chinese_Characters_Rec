from __future__ import print_function, absolute_import
import torch.utils.data as data
import numpy as np
import cv2

class _OWN(data.Dataset):
    def __init__(self, config, jpgPaths):

        self.root = config.DATASET.ROOT
        self.jpgPaths = jpgPaths
        self.inp_h = config.MODEL.IMAGE_SIZE.H
        self.inp_w = config.MODEL.IMAGE_SIZE.W

        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        print("load {} images!".format(len(jpgPaths)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.jpgPaths[idx]
        img = cv2.imread(img_path)
        text_path = img_path.replace('.jpg', '.txt')
        with open(text_path) as f:
            label = f.read().strip()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (self.inp_h, self.inp_w, 1))
        img = img.astype(np.float32)
        img = img.transpose([2, 0, 1])  # 变成（通道数，高度，宽度）
        return (img, label)








