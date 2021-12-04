# coding:utf-8
import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import argparse

"""
将训练数据图片数据，标签数据，宽高比数据存入LMDB中

"""
def init_args():
    args = argparse.ArgumentParser()
    args.add_argument('-i',
                      '--image_root_dir',
                      type=str,
                      help='The directory of the dataset , which contains the images',
                      default='train_images')
    args.add_argument('-s',
                      '--save_dir',
                      type=str
                      , help='The generated mdb file save dir',
                      default='train')
    args.add_argument('-m',
                      '--map_size',
                      help='map size of lmdb',
                      type=int,
                      default=50000000000)

    return args.parse_args()

def checkImageIsValid(imageBin):
    if imageBin is None:
        return (False, 0)
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return (False, 0)
    else:
        if imgH * imgW == 0:
            return (False, 0)
    return (True, format(imgW / imgH, '.1f'))


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k,v)

def createDataset(outputPath, imagePathList, labelList, map_size, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=map_size)
    # env = lmdb.open(outputPath)
    cache = {}
    cnt = 0
    for i in range(nSamples):
        imagePath = imagePathList[i].replace('\n', '').replace('\r\n', '')
        # print(imagePath)
        text_path = labelList[i]
        with open(text_path, 'r', encoding='utf-8') as file:
            label = file.read().replace(' ', '').replace('\r\n', '').replace('\n', '')
        # if not os.path.exists(imagePath):
        #     print('%s does not exist' % imagePath)
        #     continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            valid, ratio = checkImageIsValid(imageBin)
            if not valid:
                print('%s is not a valid image' % imagePath)
                continue
            if len(label) == 0:
                print('%s is not a valid label' % text_path)
                continue
            ratioKey = 'ratio-%09d' % cnt
            cache[ratioKey] = ratio
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt != 0 and cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    cache['num-samples'] = str(cnt)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % cnt)


def checkImgPath(path):
    return path.endswith(".jpg") or path.endswith(".JPG")

def findImages(path):
    walk = os.walk(os.path.normpath(path))
    imgs = []
    labels = []
    for path, dir_list, file_list in walk:
        for file_name in file_list:
            if checkImgPath(file_name):
                img_path = os.path.join(path, file_name)
                if img_path.endswith(".jpg"):
                    lb_path = img_path.replace('.jpg', '.txt')
                elif img_path.replace(".JPG", '.txt'):
                    lb_path = img_path.replace(".JPG", '.txt')
                else:
                    print("特殊圖片格式{}".format(img_path))
                    continue
                if not os.path.exists(lb_path):
                    print("不存在{}".format(lb_path))
                    continue
                imgs.append(img_path)
                labels.append(lb_path)
    return (imgs, labels)


def creatDB(args):
    imgDir = args.image_root_dir
    imgPathList, labelList = findImages(imgDir)
    print("picture count:{}".format(len(imgPathList)))
    createDataset(args.save_dir, imgPathList, labelList, args.map_size)

def create_label(args):
    lmdbPath = args.save_dir
    env = lmdb.open(
        lmdbPath,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False)
    label_key = 'num-samples'
    with env.begin(write=False) as txn:
       num = txn.get(label_key.encode())
    num = int(num.decode(encoding='utf-8'))
    print("樣本個數{}".format(num))
    all = set()
    for i in range(num+1):
        index = i
        with env.begin(write=False) as txn:
            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode())
            label = label.decode(encoding='utf-8')
            for char in label:
                all.add(char)
            # print("{0}: {1}".format(index, label.decode(encoding='utf-8')))
            if i % 100000 == 0:
                print(i)
    alphabets = ''.join(all)
    cur_path = os.getcwd()
    dst = os.path.join(cur_path, 'alphabets.txt')
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(alphabets)
    print("長度{}".format(len(alphabets)))

if __name__ == '__main__':
    args = init_args()
    creatDB(args)
