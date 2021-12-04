import matplotlib.pyplot as plt
import numpy as np
import torchvision

def showImage(imgs , title):
    # plt.imshow(np.asarray(imgs))
    # plt.show()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if type(imgs) is np.ndarray:
        plt.imshow(imgs, cmap='gray')
        plt.title(title)
        plt.show()
    else:
        npImg = imgs.numpy()
        plt.imshow(np.transpose(npImg, (1, 2, 0)))
        plt.title(title)
        plt.show()

def testImage(train_loader):
    dataiter = iter(train_loader)
    imgs, labels = dataiter.next()
    for i in range(32):
        if type(imgs) is np.ndarray:
          showImage(imgs[i], labels[i])
        else:
          showImage(torchvision.utils.make_grid(imgs[i]), labels[i])

