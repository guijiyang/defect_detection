# %%
import os
os.chdir('/home/guijiyang/Code/python/torch/metallic_Surface_detect.pytorch')

from unetplus import Unet_plus
from utils import rle2mask, displayTopMasks, computeDice
from ImgDataset import ImageDataset
from transform import ImageTransform
from config import detectConfig
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

data_dir = '/home/guijiyang/dataset/severstal_steel'
# %%
# defect_df=pd.read_csv('csv/defectIoU.csv')
# defect_df.head()
# #%%
# #
# small_IoU=defect_df[defect_df['diceIoU']<0.1]
# print(small_IoU)
# %%
class_df=pd.read_csv('csv/class_IoU_.csv')
class_df.head()

#%%
class_IoU=class_df[class_df['class_IoU']<0.1]
print(class_IoU.count())

# %%
cfg = detectConfig()
cfg.image_size = (1600, 256)
cfg.mean = 0.344
cfg.std = 0.14
cfg.display()
train_data = ImageDataset(data_dir, transform=ImageTransform(
    cfg.image_size, cfg.mean, cfg.std))


# %%

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

color_list=[(0.9, 0.9, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)]

# %%
with torch.no_grad():
    count = 0
    for idx in range(5):
        if count > 5:
            break
        image_id, image, masks = train_data[idx]
        # equal_image=cv2.equalizeHist(image)
        # print(equal_image)
        # fig=plt.figure(figsize=(8,16))
        # plt.imshow(equal_image,cmap='gray')
        # fig=plt.figure(figsize=(8,16))
        # plt.imshow(image,cmap='gray')
        # image_t=image_t.reshape(1,*image_t.shape)
        # image=np.expand_dims(image, axis=-1)
        # masks=np.expand_dims(masks, axis=-1)
        # image=np.repeat(image,3,axis=-1)
        # masks=np.repeat(masks, 3, axis=-1)
        # print(image.shape,masks.shape)
        image = image.reshape(1, *image.shape)
        masks = masks.reshape(1, *masks.shape)
        output = model(image.to(device))
        # pred_mask=output.cpu().clone().numpy()
        # pred_mask=np.where(pred_mask>0.5, 1., 0.)
        # pred_mask=np.transpose(pred_mask, axes=[0,2,3,1])
        # pred_mask=np.repeat(pred_mask, 3, axis=-1)
        dice_iou = computeDice(output, masks.to(device), reduction='sum')
        orignal_image = cv2.imread(os.path.join(data_dir, 'train_images', image_id))
        # orignal_image = np.repeat(orignal_image, 3, axis=-1)
        gt_masks = masks.cpu().clone().numpy()
        gt_masks = np.transpose(gt_masks, axes=[1, 2, 3, 0])
        gt_masks = np.repeat(gt_masks, 3, axis=-1)
        pred_mask = output.cpu().clone().numpy()
        pred_mask = np.transpose(pred_mask, axes=[1, 2, 3, 0])
        pred_mask = np.repeat(pred_mask, 3, axis=-1)
        displayTopMasks(orignal_image, gt_masks,
                        'id : {}, gt'.format(image_id), color_list)
        # print(class_IoU['class_IoU'][idx])
        displayTopMasks(orignal_image, pred_mask,
                        'id : {}, iou : {}'.format(image_id, dice_iou),color_list)

        # break
        count += 1


# %%
