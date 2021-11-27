from PIL import Image
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class AlignCollate(object):

    def __init__(self, config, imgH, imgW, keep_ratio=True, min_ratio=1):
        '''
        在每一批次训练之前会触发该方法
        :param imgH: 数据集高度
        :param imgW: 数据集宽度
        :param keep_ratio:
        每个批次的训练集数据宽并不一定都是相等的， 但是输入到网络中的同一批次的数据宽要定长
        '''
        self.imgH = imgH
        self.imgW = imgW
        self.config = config
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1] # 同个批次的图像的最大宽高比
            imgW = int(np.floor(max_ratio * imgH)) # 宽度取到最宽的（下面resizeNormalize会对宽度不足imgW的图片进行填充）
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        resizeNormalize = ResizeNormalize(self.config, (imgW, imgH))
        # 注意在AlignCollate中返回的是张量，dataset类中返回的可以是numpy矩阵
        # resize再转成Tensor
        images = [resizeNormalize(image) for image in images]
        # image的shape为[h,w],需要扩展为 [chanel, h, w]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

class ResizeNormalize(object):

    def __init__(self, config, size, interpolation=Image.BILINEAR):
        """
        :param size: (h, w)
        :param interpolation:
        """
        self.size = size
        self.interpolation = interpolation
        self.mean = config.DATASET.MEAN
        self.std = config.DATASET.STD

    def __call__(self, img):
        """
        :param img: PIL Image shape:(W,H)
        :return:
        """
        imgW, imgH = self.size
        # 1. 对图片等比拉伸
        scale = img.size[1]*1.0 / imgH
        w     = img.size[0] / scale
        w     = int(w)
        img = img.resize((w, imgH), self.interpolation)
        w, h  = img.size
        # 2. 若拉伸后的宽度达不到最大宽度，右边需要填充padding
        if w<=imgW:
            newImage       = np.zeros((imgH, imgW),dtype='uint8')
            newImage[:]    = 255
            newImage[:, :w] = np.array(img)
            img            = Image.fromarray(newImage)
        else:
            img = img.resize((imgW, imgH), self.interpolation)
        img = transforms.ToTensor()(img)
        img.sub_(self.mean).div_(self.std)
        return img