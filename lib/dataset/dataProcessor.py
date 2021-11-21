import os

def checkImgPath(path):
    return path.endswith(".jpg") or path.endswith(".JPG")

def findImage(path):
    walk = os.walk(path)
    res = []
    for path, dir_list, file_list in walk:
        for file_name in file_list:
            if checkImgPath(file_name):
                print(path + "/" + file_name)
                res.append(path + "/" + file_name)
    return res

def getData(config):
    root = config.DATASET.ROOT
    base_dir = config.DATASET.BASE_DIR
    res = []
    for d in base_dir:
        r = findImage(root + d)
        res += r
    print("find: {0} image".format(len(res)))