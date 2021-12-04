import os
import yaml
import pandas as pd
import lmdb
from easydict import EasyDict as edict

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

if __name__ == '__main__':
    print("xxxx")
    lmdb_path = "C:\lmdb"
    findRatio(lmdb_path)
