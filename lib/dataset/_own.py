from __future__ import print_function, absolute_import
import torch.utils.data as data
import numpy as np
from PIL import Image
import lmdb
import six

class _OWN(data.Dataset):
    def __init__(self, config, indexs):
        self.env = lmdb.open(
            config.DATASET.LMDB_PATH,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        self.dataset_name = config.DATASET.DATASET
        self.config = config
        self.indexs = indexs

    def __len__(self):
        return len(self.indexs)

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % idx
            image_key = 'image-%09d' % idx
            ration_key = 'ratio-%09d' % idx

            label = txn.get(label_key.encode())
            label = label.decode(encoding='utf-8')
            label = ''.join([x for x in label if x in self.config.DATASET.ALPHABETS])

            imgbuf = txn.get(image_key.encode('utf-8'))
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            image = Image.open(buf).convert('L')

            ratio = txn.get(ration_key.encode('utf-8'))
        return (image, label, float(ratio.decode(encoding='utf-8')))

    def get_image_ratio(self, idx):
        with self.env.begin(write=False) as txn:
             ration_key = 'ratio-%09d' % idx
             ratio = txn.get(ration_key.encode('utf-8'))
        return float(ratio.decode(encoding='utf-8'))




