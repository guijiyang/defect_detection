# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'torch/metallic_Surface_detect.pytorch'))
	print(os.getcwd())
except:
	pass

#%%
import os
import sys
sys.path.insert(0, '/home/guijiyang/Code/python/torch/metallic_Surface_detect.pytorch')
# os.chdir('/home/guijiyang/Code/python/torch/metallic_Surface_detect.pytorch')

from unetplus import Unet_plus
from utils import rle2mask, displayTopMasks, computeDice
from ImgDataset import ImageDataset
from transform import ImageTransform
from config import detectConfig
from detect import postMask
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
data_dir = '/home/guijiyang/dataset/severstal_steel'


#%%
class_df=pd.read_csv('csv/class_IoU_.csv')

class_IoU=class_df[class_df['class_IoU']<0.1]
class_IoU.head()


#%%
cfg = detectConfig()
cfg.image_size = (1600, 256)
cfg.mean = 0.344
cfg.std = 0.14
cfg.min_size=1000
cfg.display()
train_data = ImageDataset(data_dir, transform=ImageTransform(
    cfg.image_size, cfg.mean, cfg.std))



device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

WEIGHT_PATH = 'weights'
MODEL_NAME = os.path.join(WEIGHT_PATH, 'unet_plus_99.pth')

# 加载模型
model = Unet_plus(1, 4).to(device)

if os.path.exists(MODEL_NAME):
    model.load_state_dict(torch.load(MODEL_NAME))
else:
    raise Exception('cannot find model weights')

model.eval()


#%%
with torch.no_grad():
    count = 0
    for idx in class_IoU.index:
        if count > 5:
            break
        print(idx)
        image_id, image, masks = train_data[idx]
        image = image.reshape(1, *image.shape)
        masks = masks.reshape(1, *masks.shape)
        output = model(image.to(device))
        
        dice_iou = computeDice(output, masks.to(device), reduction='sum')
        
        orignal_image = cv2.imread(os.path.join(data_dir, 'train_images', image_id))
        gt_masks = masks.cpu().clone().numpy()
        gt_masks = np.transpose(gt_masks, axes=[1, 2, 3, 0])
        gt_masks = np.repeat(gt_masks, 3, axis=-1)
        pred_mask = output.cpu().clone().numpy()
        pred_mask,num=postMask(pred_mask, cfg.threshold, cfg.min_size)
        print(num)
        pred_mask = np.transpose(pred_mask, axes=[1, 2, 3, 0])
        pred_mask = np.repeat(pred_mask, 3, axis=-1)
        displayTopMasks(orignal_image, gt_masks,
                        'id : {}, gt'.format(image_id))
        displayTopMasks(orignal_image, pred_mask,
                        'id : {}, iou : {}'.format(image_id, dice_iou))

        # break
        count += 1


