import os.path

import numpy as np
import tensorflow as tf
import json

import torch
from torch.utils.data import Dataset

with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
"""
不需要归一化，npy文件存储的像素就是[-1,1]
root: 数据集根路径，下面有三个文件夹，分别是images,images_npy,labels
"""
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.imgs_list = os.listdir(os.path.join(root, 'images_npy'))
        self.labels_list = os.listdir(os.path.join(root, 'labels'))
        self.transform = transform

    def __getitem__(self, index):
        img = np.load(os.path.join(self.root,'images_npy',self.imgs_list[index]))
        img = torch.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)
        label = np.load(os.path.join(self.root,'labels',self.labels_list[index]))
        label = torch.from_numpy(label)
        return img, label

    def __len__(self):
        return len(self.imgs_list)