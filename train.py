import argparse
from easydict import EasyDict as edict
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler
import lib.models.crnn as crnn
import lib.utils.utils as utils
from lib.core import function
import lib.config.alphabets as alphabets
import lib.config.keys as alphabetsKeys
from lib.utils.utils import model_info
from lib.dataset.AlignCollate import AlignCollate
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from lib.dataset.DynamicBatchSampler import DynamicBatchSampler
from lib.dataset._own import _OWN
import random
from torch_baidu_ctc import CTCLoss

def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    with open(config.DATASET.ALPHABETS, 'r', encoding='utf-8') as file:
        config.DATASET.ALPHABETS = file.read().replace(' ', '').replace('\r\n', '').replace('\n', '')
    # config.DATASET.ALPHABETS = alphabets.alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    print("字符表个数{}".format(config.MODEL.NUM_CLASSES))
    return config

def main():

    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    model = crnn.get_crnn(config)

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    # criterion = CTCLoss(reduction='mean')

    if config.CUDNN.CTCENABLE:
       criterion = criterion.to(device)

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    # if isinstance(config.TRAIN.LR_STEP, list):
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, config.TRAIN.LR_STEP,
    #         config.TRAIN.LR_FACTOR, last_epoch-1
    #     )
    # else:
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #         optimizer, config.TRAIN.LR_STEP,
    #         config.TRAIN.LR_FACTOR, last_epoch - 1
    #     )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        from collections import OrderedDict
        model_dict = OrderedDict()
        for k, v in checkpoint.items():
            if 'cnn' in k:
                model_dict[k[4:]] = v
        model.cnn.load_state_dict(model_dict)
        if config.TRAIN.FINETUNE.FREEZE:
            for p in model.cnn.parameters():
                p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)

    # 數據
    num_samples = utils.get_num_samples(config.DATASET.LMDB_PATH)
    indexs = list(range(0, num_samples))

    trainIdxs, testIndexs = train_test_split(indexs, test_size=0.1, random_state=1024)  ##此处未考虑字符平衡划分
    # random.shuffle(trainIdxs)
    trainIdxs = trainIdxs[:5500000]
    train_sampler = RandomSampler(trainIdxs)
    test_sampler = RandomSampler(testIndexs)

    train_dataset = _OWN(config, trainIdxs)
    val_dataset = _OWN(config, testIndexs)

    # padding 比例不高于10%,  max_tokens = batch_size * ratio = 100 * 5 即假设图片为32 * 160 batch 100
    train_sampler = DynamicBatchSampler(train_sampler, train_dataset.get_image_ratio, num_buckets=120, min_size=2, max_size=37,
                                          max_tokens=500, max_sentences=100)
    val_sampler = DynamicBatchSampler(test_sampler, val_dataset.get_image_ratio, num_buckets=120, min_size=2, max_size=37,
                                        max_tokens=500, max_sentences=100)
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=AlignCollate(config, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W),
        batch_sampler=train_sampler
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=AlignCollate(config, config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W),
        batch_sampler=val_sampler,
    )

    # from testUnit.testUnit import testImage
    # testImage(train_loader)
    # return
    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)


    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):

        function.train(config, train_loader, train_dataset, converter, model, criterion, optimizer, device, epoch, writer_dict)
        # lr_scheduler.step()

        acc = function.validate(config, val_loader, val_dataset, converter, model, criterion, device, epoch, writer_dict, output_dict)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)
        # save checkpoint
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            },  os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )
    writer_dict['writer'].close()

if __name__ == '__main__':

    main()