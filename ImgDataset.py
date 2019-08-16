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
        transform: transform for Images.
    """

    def __init__(self, dataset_dir, transform=None):
        self.transform = transform
        self.image_infos = []
        train_fd = pd.read_csv(os.path.join(dataset_dir, 'train.csv'))
        total_count = train_fd['ImageId_ClassId'].count()
        image_info = None
        cur_image_id = None
        for idx in range(total_count):
            image_id, class_id = train_fd['ImageId_ClassId'][idx].split('_')
            if cur_image_id != image_id:
                if image_info is not None:
                    self.image_infos.append(image_info)
                cur_image_id = image_id
                image_info = {
                    'ImageId_path': os.path.join(dataset_dir, 'train_images', cur_image_id),
                    'class_id': [],
                    'mask': []}
            if pd.notnull(train_fd['EncodedPixels'][idx]):
                image_info['class_id'].append(int(class_id))
                image_info['mask'].append(train_fd['EncodedPixels'][idx])
        self.image_infos.append(image_info)

    def __len__(self):
        return len(self.image_infos)

    def __getitem__(self, index):
        image_info = self.image_infos[index]
        image = cv2.imread(image_info['ImageId_path'])
        image_shape = image.shape
        masks = np.zeros((*image_shape[:-1],4), dtype=np.uint8)
        for idx in range(len(image_info['mask'])):
            mask = rle2mask(image_info['mask'][idx], image_shape[:-1])
            masks[..., image_info['class_id'][idx]-1] = mask
        masks = np.clip(masks, 0, 1)
        if self.transform:
            image, masks = self.transform(image, masks)
        return image_info['ImageId_path'].split('/')[-1], image, masks


class_id_map = {'1': 0, '2': 1, '3': 2, '4': 3}


class MaskDataset(data.Dataset):
    r"""
    Mask dataset for train and test
    """

    def __init__(self, dataset_dir,  split_ratio=0.9, transform=None):
        self.transform = transform
        self.image_infos = []
        self.image_count = 0
        fd = pd.read_csv(os.path.join(dataset_dir, 'mask.csv'),
                         names=['image_id', 'class_id'])
        total_count = fd.image_id.count()
        for idx in range(total_count):
            image_id = fd.image_id[idx]
            class_id = class_id_map[str(fd.class_id[idx])]
            self.image_infos.append({'image_path': os.path.join(
                dataset_dir, 'mask', image_id), 'class_id': class_id})
        self.image_count = len(self.image_infos)

    def __len__(self):
        return self.image_count

    def __getitem__(self, index):
        image_info = self.image_infos[index]
        image = cv2.imread(image_info['image_path'], 0)
        class_id = image_info['class_id']
        if self.transform:
            image = self.transform(image)
        return image, class_id
