import os
import pandas as pd
import lmdb
import numpy as np
import cv2
import random
from tqdm import tqdm
import six
from PIL import Image

def findRatio(lmdb_path):

    env = lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)
    key = 'num-samples'
    with env.begin(write=False) as txn:
        num = txn.get(key.encode())

    num = int(num.decode(encoding='utf-8'))
    ratios = []
    for i in range(0, num + 1):
        ratioKey = 'ratio-%09d' % i
        with env.begin(write=False) as txn:
            ratio = txn.get(ratioKey.encode())
        ratio = float(ratio.decode(encoding='utf-8'))
        ratios.append(ratio)
    env.close()
    print("=====20======")
    r = pd.cut(ratios, 20)
    print(pd.value_counts(r))
    print("=====10======")
    r = pd.cut(ratios, 10)
    print(pd.value_counts(r))


def compute_std_mean(lmdb_path, NUM=None):
    imgs = np.zeros([32, 160, 1, 1])
    means, stds = [], []
    env = lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)
    key = 'num-samples'
    with env.begin(write=False) as txn:
        num = txn.get(key.encode())

    num = int(num.decode(encoding='utf-8'))
    indexs = list(range(0, num))
    print("总共有{}张图片, shuffle中".format(len(indexs)))
    random.shuffle(indexs)
    print("shuffle完毕，抽样{}张图片计算均值和方差".format(NUM))
    with env.begin(write=False) as txn:
        for i in tqdm(range(NUM)):
            idx = indexs[i]
            image_key = 'image-%09d' % idx
            imgbuf = txn.get(image_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            imageBuf = np.frombuffer(imgbuf, dtype=np.uint8)
            img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape[:2]
            img = cv2.resize(img, (0, 0), fx=160 / w, fy=32 / h, interpolation=cv2.INTER_CUBIC)
            img = img[:, :, np.newaxis, np.newaxis]
            imgs = np.concatenate((imgs, img), axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(1):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()
    # print(means, stds)
    env.close()
    return stds, means
if __name__ == '__main__':
    lmdb_path = "C:\lmdb"
    # findRatio(lmdb_path)
    stds, means = compute_std_mean(lmdb_path, NUM=10000)
    print("stds = ", stds, "means = ", means)