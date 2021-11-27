import matplotlib.pyplot as plt
import numpy as np
import torchvision

def showImage(imgs , title):
    # plt.imshow(np.asarray(imgs))
    # plt.show()
    npImg = imgs.numpy()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.imshow(np.transpose(npImg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def testImage(train_loader):
    dataiter = iter(train_loader)
    imgs, labels = dataiter.next()
    # imgs = imgs.squeeze(1)
    # imgs = imgs.numpy()
    # showImage(imgs)
    for i in range(32):
        showImage(torchvision.utils.make_grid(imgs[i]), labels[i])