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


class CropDataset(data.Dataset):
    r"""
    croped 256x256 image dataset
    """

    def __init__(self, dataset_dir, csv_path, transform=None):
        self.transform = transform
        self.image_infos = []
        self.image_count = 0
        image_p_path = os.path.join(dataset_dir, 'images')
        image_n_path = os.path.join(dataset_dir, 'images_n')
        # mask_path = os.path.join(dataset_dir, 'masks')
        img_p_list = next(os.walk(image_p_path))[2]
        img_n_list = list(pd.read_csv(os.path.join(
            csv_path, 'pred.csv')).head(12000).fname)
        img_list = img_p_list+img_n_list
        img_list = sorted(img_list)
        for img_id in img_list:
            if img_id in img_p_list:
                self.image_infos.append({
                    'image_path': os.path.join(image_p_path, img_id), 'mask_path': os.path.join(dataset_dir, 'masks', img_id)
                })
            else:
                self.image_infos.append({
                    'image_path': os.path.join(image_n_path, img_id), 'mask_path': None
                })

        self.image_count = len(self.image_infos)

    def __len__(self):
        return self.image_count

    def __getitem__(self, index):
        image_info = self.image_infos[index]
        image = cv2.imread(image_info['image_path'])
        image_shape = image.shape
        masks = np.zeros((*image_shape[:-1], 4), dtype=np.uint8)
        if image_info['mask_path']:
            mask=cv2.imread(image_info['mask_path'],0)
            for idx in range(4):
                class_id=idx+1
                masks[...,idx]=np.where(mask==class_id,1,0)
        if self.transform:
            image, masks = self.transform(image, masks)
        return image_info['image_path'].split('/')[-1], image, masks