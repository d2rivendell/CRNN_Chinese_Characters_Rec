from __future__ import print_function, absolute_import
import torch.utils.data as data
import numpy as np
import cv2

class _OWN(data.Dataset):
    def __init__(self, config, jpgPaths):

        self.root = config.DATASET.ROOT
        self.jpgPaths = jpgPaths
        self.dataset_name = config.DATASET.DATASET

        self.mean = np.array(config.DATASET.MEAN, dtype=np.float32)
        self.std = np.array(config.DATASET.STD, dtype=np.float32)

        char_file = config.DATASET.CHAR_FILE
        with open(char_file, 'rb') as file:
            self.char_dict = {num: char.strip().decode('gbk', 'ignore') for num, char in enumerate(file.readlines())}
        self.label = ""
        print("load {} images!".format(len(jpgPaths)))

    def __len__(self):
        return len(self.jpgPaths)

    def __getitem__(self, idx):
        img_path = self.jpgPaths[idx]
        img = cv2.imread(img_path)
        text_path = img_path.replace('.jpg', '.txt')
        with open(text_path, 'r', encoding='utf-8') as file:
            contents = file.readlines()[0]
            self.label = contents
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        return (img, self.label)








