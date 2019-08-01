import os
import cv2
import matplotlib.pyplot as plt
import torch.utils.data as data
import pandas as pd
import numpy as np

from utils import rle2mask


class ImageDataset(data.Dataset):
    r"""
    Image Dataset use for train or test

    Args:
        dataset_dir: directory of dataset;
        mode: 'train' or 'test' mode;
        split_ratio: ratio of train/test data split;
        transform: transform for Images.
    """

    def __init__(self, dataset_dir, mode='train', split_ratio=0.7, transform=None):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.split_ratio=split_ratio
        self.transform = transform
        self.image_infos = []
        train_fd = pd.read_csv(os.path.join(self.dataset_dir, 'train.csv'))
        self.total_count=train_fd['ImageId_ClassId'].count()
        if self.mode == 'train':
            self.img_count=int(self.total_count*self.split_ratio)
            image_info = None
            cur_image_id = None
            for idx in range(self.img_count):
                image_id = train_fd['ImageId_ClassId'][idx].split('_')[0]
                if cur_image_id != image_id:
                    if image_info != None:
                        self.image_infos.append(image_info)
                    cur_image_id = image_id
                    image_info = {
                        'ImageId_path': os.path.join(self.dataset_dir, 'train', cur_image_id),
                        'mask': [None if pd.isnull(train_fd['EncodedPixels'][idx]) else train_fd['EncodedPixels'][idx]]}
                else:
                    image_info['mask'].append(None if pd.isnull(train_fd['EncodedPixels'][idx]) else train_fd['EncodedPixels'][idx])
            self.image_infos.append(image_info)
        else:
            self.img_count=self.total_count-int(self.total_count*self.split_ratio)
            image_info = None
            cur_image_id = None
            for idx in range(self.total_count-1, self.total_count-self.img_count, -1):
                image_id = train_fd['ImageId_ClassId'][idx].split('_')[0]
                if cur_image_id != image_id:
                    if image_info != None:
                        self.image_infos.append(image_info)
                    cur_image_id = image_id
                    image_info = {
                        'ImageId_path': os.path.join(self.dataset_dir, 'train', cur_image_id),
                        'mask': [None if pd.isnull(train_fd['EncodedPixels'][idx]) else train_fd['EncodedPixels'][idx]]}
                else:
                    image_info['mask'].append(None if pd.isnull(train_fd['EncodedPixels'][idx]) else train_fd['EncodedPixels'][idx])
            self.image_infos.append(image_info)
            

    def __len__(self):
        return self.img_count

    def __getitem__(self, index):
        image_info = self.image_infos[index]
        image = cv2.imread(image_info['ImageId_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image=np.expand_dims(image,axis=-1)
        masks = np.zeros_like(image,dtype=np.uint8)
        for idx in range(len(image_info['mask'])):
            # if image_info['mask'][idx]!= None:
            mask=rle2mask(image_info['mask'][idx], image.shape[:2])
            mask=np.expand_dims(mask, axis=-1)
            mask=np.repeat(mask, image.shape[2], axis=-1)
            masks+=mask
        masks=np.clip(masks,0,1)
        if self.transform:
            image, masks = self.transform(image, masks)
        return image, masks
