import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
# from lib.dataset.dataProcessor import findImage
import yaml
from easydict import EasyDict as edict
import argparse
import os
 
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='C:/Users/fander/Desktop/GitHub/CRNN_Chinese_Characters_Rec/lib/config/OWN_config.yaml')
    parser.add_argument('--image_path', type=str, default='C:/Users/fander/Desktop/crnn_test_images', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='C:/Users/fander/Desktop/GitHub/CRNN_Chinese_Characters_Rec/output/OWN/crnn/2021-12-11-10-27(5)/checkpoints/checkpoint_5_acc_0.5239.pth',
                        help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    with open(config.DATASET.ALPHABETS, 'r', encoding='utf-8') as file:
        config.DATASET.ALPHABETS = file.read().replace(' ', '').replace('\r\n', '').replace('\n', '')
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args

def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)


    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.astype(np.float32)

    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, 1, *img.size())
    model.eval()
    preds = model(img)
    # print(preds.shape)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def checkImgPath(path):
    return path.endswith(".jpg") or path.endswith(".png")

def findImage(path):
    walk = os.walk(os.path.normpath(path))
    res = []
    for path, dir_list, file_list in walk:
        for file_name in file_list:
            if checkImgPath(file_name):
                res.append(os.path.join(path, file_name))
    return res
if __name__ == '__main__':

    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = crnn.get_crnn(config).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    started = time.time()

    pahts = findImage(args.image_path)
    for path in pahts:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        res = recognition(config, img, model, converter, device)
        print('{0}:{1}'.format(os.path.split(path)[1], res))
    finished = time.time()
    print('elapsed time: {0}'.format((finished - started)/len(pahts)))


