from  __future__ import  absolute_import
import time
import lib.utils.utils as utils
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(config, train_loader, dataset, converter, model, criterion, optimizer, device, epoch, writer_dict=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    end = time.time()
    batchs = 0
    epoch_begin = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time() - end)
        images = images.to(device)
        # inference
        if config.CUDNN.CTCENABLE:
            preds = model(images)
        else:
            preds = model(images).cpu()
        # compute loss(device)
        batch_size = images.size(0)
        text, length = converter.encode(labels)  # length = 一个batch中的总字符长度, text = 一个batch中的字符所对应的下标
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)  # timestep * batchsize
        # if config.CUDNN.CTCENABLE:
        # text = text.to(device)
        # length = length.to
        # preds_size = preds_size.to(device)
        loss = criterion(preds, text, preds_size, length)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

        batch_time.update(time.time() - end)
        batchs += len(images)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, batchs, len(dataset), batch_time=batch_time,
                speed=images.size(0) / batch_time.val,
                data_time=data_time, loss=losses)
            print(msg)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.avg, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1
        end = time.time()
    print("========== epoch  time:{0} min ==============".format((time.time()-epoch_begin)/60))

def validate(config, val_loader, dataset, converter, model, criterion, device, epoch,  writer_dict, output_dict):

    losses = AverageMeter()
    model.eval()

    n_correct = 0
    batchs = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):

            images = images.to(device)
            # inference
            if config.CUDNN.CTCENABLE:
                preds = model(images)
            else:
                preds = model(images).cpu()

            # compute loss
            batch_size = images.size(0)
            text, length = converter.encode(labels)

            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            # if config.CUDNN.CTCENABLE:
            #     text = text.to(device)
            #     length = length.to(device)
            #     preds_size = preds_size.to(device)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), images.size(0))

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1
            batchs += len(images)
            if (i + 1) % config.PRINT_FREQ == 0:
                print('Epoch: [{0}][{1}/{2}]'.format(epoch, batchs, len(dataset)))

            if i == config.TEST.NUM_TEST_BATCH:
                break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    num_test_sample = batchs
    if num_test_sample > len(dataset):
        num_test_sample = len(dataset)

    print("[#correct:{} / #total:{}]".format(n_correct, num_test_sample))
    accuracy = n_correct / float(num_test_sample)
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy