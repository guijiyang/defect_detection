#%%
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import os
os.chdir('/home/guijiyang/Code/python/torch/metallic_Surface_detect.pytorch')

from utils import rle2mask,displayTopMasks
from ImgDataset import ImageDataset
from unetplus import Unet_plus
from detect import DetectNet
data_dir='/home/guijiyang/dataset/severstal_steel'
#%%
defect_df=pd.read_csv('defectIoU.csv')
defect_df.head()
#%%
# 
small_IoU=defect_df[defect_df['diceIoU']<0.1]
print(small_IoU)
#%%
class_df=pd.read_csv('class_IoU.csv')
class_df.head()

#%%
class_IoU=class_df[class_df['class_IoU']<0.1]
print(class_IoU.count())
#%%
train_data=ImageDataset(data_dir)

#%%


#%%

MODEL_DIR='weights'
CASCADE_PATH=os.path.join(MODEL_DIR, 'unet_first_100.pth')
CLASSIFIER_PATH=os.path.join(MODEL_DIR, 'compactNet.pth')

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
model=Unet_plus(1, 1).to(device)

if os.path.exists(CASCADE_PATH):
    cascade_dict=torch.load(CASCADE_PATH,  map_location=lambda storage, loc: storage)
else:
    raise Exception('cannot find model weights')

if os.path.exists(CLASSIFIER_PATH):
    classifier_dict=torch.load(CLASSIFIER_PATH,  map_location=lambda storage, loc: storage)
else:
    raise Exception('cannot find model weights')

model.load_state_dict(cascade_dict)

model.eval()

toTensor=transforms.ToTensor()

#%%
with torch.no_grad():
    count=0
    for idx in small_IoU.index:
        if count>5:
            break
        image_id, image, masks=train_data[idx]
        equal_image=cv2.equalizeHist(image)
        # print(equal_image)
        fig=plt.figure(figsize=(8,16))
        plt.imshow(equal_image,cmap='gray')
        # fig=plt.figure(figsize=(8,16))
        # plt.imshow(image,cmap='gray')
        image_t=toTensor(image)
        image_t=image_t.reshape(1,*image_t.shape)
        image=np.expand_dims(image, axis=-1)
        masks=np.expand_dims(masks, axis=-1)
        image=np.repeat(image,3,axis=-1)
        masks=np.repeat(masks, 3, axis=-1)
        # print(image.shape,masks.shape)
        output=model(image_t.to(device))
        pred_mask=output.cpu().clone().numpy()
        pred_mask=np.where(pred_mask>0.5, 1., 0.)
        pred_mask=np.transpose(pred_mask, axes=[0,2,3,1])
        pred_mask=np.repeat(pred_mask, 3, axis=-1)
        displayTopMasks(image, masks,'id : {}, gt'.format(image_id))
        displayTopMasks(image, pred_mask,'id : {}, pred'.format(image_id))
        print(image_id)
        break
        count+=1


#%%
