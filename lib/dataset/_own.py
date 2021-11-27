from __future__ import print_function, absolute_import
import torch.utils.data as data
import numpy as np
from PIL import Image

class _OWN(data.Dataset):
    def __init__(self, config, jpgpaths):
        self.root = config.DATASET.ROOT
        self.jpgpaths = jpgpaths
        self.dataset_name = config.DATASET.DATASET
        self.config = config
        print("load {} images!".format(len(jpgpaths)))

    def __len__(self):
        return len(self.jpgpaths)

    def __getitem__(self, idx):
        img_path = self.jpgpaths[idx]
        img = Image.open(img_path).convert('L')
        text_path = img_path.replace('.jpg', '.txt')
        with open(text_path, 'r', encoding='utf-8') as file:
            label = file.readlines()[0].replace(' ', '')
        # 排除掉训练集中存在一些不在字符表中生僻字
        label = ''.join([x for x in label if x in self.config.DATASET.ALPHABETS])
        assert (len(label) > 0)
        return (img, label)








