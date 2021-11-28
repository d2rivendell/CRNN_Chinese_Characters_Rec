'''
360万中文训练集标签修改
'''

# # chinese characters dictionary for 3.6 million data set.
# with open('../char_std_5990.txt', 'rb') as file:
# 	char_dict = {num : char.strip().decode('gbk','ignore') for num, char in enumerate(file.readlines())}
#
# # processing output
# with open('../test.txt') as file:
# 	value_list = ['%s %s'%(segment_list.split(' ')[0], ''.join([char_dict[int(val)] for val in segment_list[:-1].split(' ')[1:]])) for segment_list in file.readlines()]
#
# # final output
# with open('test.txt', 'w', encoding='utf-8') as file:
# 	[file.write(val+'\n') for val in value_list]

'''
orginal version
'''

# with open('../char_std_5990.txt', 'rb') as file:
# 	char_dict = {num : char.strip().decode('gbk','ignore') for num, char in enumerate(file.readlines())}

#  value_list = []
# with open('../test.txt') as file:
# 	label_list = file.readlines()
# 	for segment_list in label_list:
# 		key = segment_list.split(' ')[0]
# 		segment_list = segment_list[:-1].split(' ')[1:]
# 		temp = [char_dict[int(val)] for val in segment_list]
# 		value_list.append('%s %s'%(key, ''.join(temp)))

# with open('test.txt', 'w', encoding='utf-8') as file:
# 	[ file.write(val+'\n') for val in value_list]

import pandas as pd
from PIL import Image
import os


def createTrainAlphabetsPath():
    cur_path = os.getcwd()
    dst = os.path.join(cur_path, 'lib', 'config')
    print(dst)
    if not os.path.exists(dst):
        os.makedirs(dst) # mkdir只能建一個目錄， 多級目錄使用makedirs
    return dst

def createOtherAlphabetsIfNeed(dataset_name, words):
    if len(words) > 0:
        dst = createTrainAlphabetsPath()
        text_path = os.path.join(dst, dataset_name + '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(words)


def preparedata(jpgpaths, config, add_new_chars=False):
    """
    預處理數據,查看圖片寬高占比分佈，新增數據集中的生僻字到字符表
    :param jpgpaths:
    :param config:
    :param add_new_chars: 是否將訓練集的生僻字放到字符表中訓練
    :return:
    """
    dataset_name = config.DATASET.DATASET
    alphabets = config.DATASET.ALPHABETS
    length = len(jpgpaths)
    ratios = []
    no_exits_chars = set()
    for i in range(length):
        # 統計比例
        img_path = jpgpaths[i]
        img = Image.open(img_path).convert('L')
        W, H = img.size
        ratios.append(W * 1.0 / H)
        # 文字
        text_path = img_path.replace('.jpg', '.txt')
        with open(text_path, 'r', encoding='utf-8') as file:
            label = file.readlines()[0].replace(' ', '')
            # 排除掉训练集中存在一些不在字符表中生僻字
            for x in label:
                if x not in alphabets:
                   no_exits_chars.add(x)
    r = pd.cut(ratios, 10)
    print(pd.value_counts(r))
    if add_new_chars:
        words = ''.join(no_exits_chars)
        words = words.replace(' ', '')
        createOtherAlphabetsIfNeed(dataset_name, words)
        l = len(words)
        if l > 0:
            print("原字符個數{}".format(len(config.DATASET.ALPHABETS)))
            print("新增字符個數{0}".format(l))
            config.DATASET.ALPHABETS += words
            print("新增后 len: {}".format(len(config.DATASET.ALPHABETS)))


def excludeImage(jpgpaths, ratio):
    """
    抛棄寬高比超過ratio以上的圖片，這個數值可以通過上面preparedata方法查案數據分佈
    :param ratio:
    :return:
    """
    length = len(jpgpaths)
    res = []
    for i in range(length):
        # 統計比例
        img_path = jpgpaths[i]
        img = Image.open(img_path).convert('L')
        W, H = img.size
        r = (W * 1.0 / H)
        if r < ratio:
           res.append(img_path)
        else:
            print(img_path)
    print("原長度{0}，排除后剩餘{1}".format(length, len(res)))
    return res