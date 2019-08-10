
from ImgDataset import ImageDataset, class_id_map
from detect import DetectNet
from utils import mask2rle
# from unetplus import Unet_plus
# from classifier import CompactNet
import os
path = os.path.abspath(__file__)
os.chdir(os.path.dirname(path))
import torch
import cv2
import numpy as np
import pandas as pd

DATA_DIR='/home/guijiyang/dataset/severstal_steel/test_images'
MODEL_DIR='weights'
CASCADE_PATH=os.path.join(MODEL_DIR, 'unet_first_100.pth')
CLASSIFIER_PATH=os.path.join(MODEL_DIR, 'compactNet.pth')
test_images=next(os.walk(DATA_DIR))[2]

device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
model=DetectNet(0.5, 3500).to(device)

if os.path.exists(CASCADE_PATH):
    cascade_dict=torch.load(CASCADE_PATH,  map_location=lambda storage, loc: storage)
else:
    raise Exception('cannot find model weights')

if os.path.exists(CLASSIFIER_PATH):
    classifier_dict=torch.load(CLASSIFIER_PATH,  map_location=lambda storage, loc: storage)
else:
    raise Exception('cannot find model weights')

model.load_state_dict(cascade_dict, classifier_dict)

model.eval()

submission = []
with torch.no_grad():
    for image_path in test_images:
        image=cv2.imread(os.path.join(DATA_DIR,image_path),0)
        # images=np.expand_dims(image, axis=(0,1))
        images_t=torch.as_tensor(image, dtype=torch.float32).reshape(1,1,*image.shape).to(device)
        output=model(images_t)
        if output is not None:
            for idx,mask in enumerate(output[:]):
                str_run_length=mask2rle(mask)
                image_id=image_path+'_'+str(idx+1)
                submission.append([image_id, str_run_length])
            
# Save to csv file
df=pd.DataFrame(submission, columns=['ImageId_ClassId','EncodedPixels'])
df.to_csv('submission.csv', index=False)