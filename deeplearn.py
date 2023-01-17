
#独属于VOC2012数据集

import collections
import hashlib
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
from tqdm import tqdm
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
d2l = sys.modules[__name__]
import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = \
    ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def read_voc_images(voc_dir, is_train=True):
    """
    :param voc_dir:VOC数据集的位置，解压后的位置可能在：.../VOCdevkit\VOC2012
    :param is_train:Ture是训练集
    :return:特征与标签
    """
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(
            torchvision.io.read_image(
                os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'),
                mode))
    return features, labels

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """
    :param imgs:
    :param num_rows:
    :param num_cols:
    :param titles:
    :param scale:
    :return:
    :example:
            n = 5
            imgs = train_features[0:n] + train_labels[0:n]
            imgs = [img.permute(1,2,0) for img in imgs]  #画的时候将通道放在后面
            d2l.show_images(imgs, 2, n);
    """
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def voc_colormap2label():
    """
    :return: 构建从RGB到VOC类别索引的映射
    """
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label


def voc_label_indices(colormap, colormap2label):
    """
    :param colormap:
    :param colormap2label:voc_colormap2label()
    :return:将VOC标签中的RGB值映射到它们的类别索引
    """
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 +
           colormap[:, :, 2])
    return colormap2label[idx]


def download_file(url):
    """
    :param url:下载文件所在url链接
    :return: 下载的位置处于根目录
    """
    print("------", "Start download with urllib")
    name = url.split("/")[-1]
    resp = requests.get(url, stream=True)
    content_size = int(resp.headers['Content-Length']) / 1024  # 确定整个安装包的大小
    # 下载到上一级目录
    path = os.path.abspath(os.path.dirname(os.getcwd())) + "\\" + name
    # 下载到该目录
    path = os.getcwd() + "\\" + name
    print("File path:  ", path)
    with open(path, "wb") as file:
        print("File total size is:  ", content_size)
        for data in tqdm(iterable=resp.iter_content(1024), total=content_size, unit='k', desc=name):
            file.write(data)
    print("------", "finish download with urllib\n\n")

def load_data_voc(batch_size, crop_size):
    """
    下载并读取Pascal VOC2012语义分割数据集
    :param batch_size: 批量大小
    :param crop_size: 裁剪大小,比如(320, 480)
    :return: 返回训练集和测试集的数据迭代器
    """
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


def voc_rand_crop(feature, label, height, width):
    """
    :param feature:
    :param label:
    :param height:
    :param width:
    :return: 随机裁剪特征和标签图像
    :example:
            imgs = []
            for _ in range(n):
                imgs += d2l.voc_rand_crop(train_features[0], train_labels[0], 200, 300)

            imgs = [img.permute(1, 2, 0) for img in imgs]
            d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
    """
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""
    def __init__(self, is_train, crop_size, voc_dir):
        """可以查看训练集和测试集所保留的样本个数"""
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        """对输入图像的RGB三个通道的值分别做标准化"""
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        """由于数据集中有些图像的尺寸可能小于随机裁剪所指定的输出尺寸，这些样本可以通过自定义的filter函数移除掉"""
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        """__getitem__函数，我们可以任意访问数据集中索引为idx的输入图像及其每个像素的类别索引"""
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)














