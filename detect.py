import pandas as pd
import numpy as np
import cv2
import torch.utils.data as data
import torch
from config import detectConfig
from ImgDataset import ImageDataset, class_id_map
from utils import mask2rle
from unetplus import Unet_plus
from transform import ImageTransform
import os
path = os.path.abspath(__file__)
os.chdir(os.path.dirname(path))


def detection_collate(batch):
    img_path = []
    imgs = []
    for sample in batch:
        img_path.append(sample[0])
        imgs.append(sample[1])
    return img_path, torch.stack(imgs, 0)


def postMask(pred, threshold,  min_size):
    pred = np.where(pred>threshold,1,0).astype(np.uint8)
    pred_mask=np.zeros_like(pred)
    num=0
    for i,masks in enumerate(pred):
        for j,mask in enumerate(masks):
            num_component, component = cv2.connectedComponents(mask)
            for idx in range(1, num_component):
                points = (component == idx)
                if points.sum() > min_size:
                    pred_mask[i,j,points]=1
                    # points_value=np.where(points)
                    # points_value=np.stack([points_value[0],points_value[1]], axis=1)
                    # for point in points_value:
                    #     pred_mask[i,j,point[0],point[1]]=1
                    num+=1
    return pred_mask, num


class EvalImageDataset(data.Dataset):
    def __init__(self, dataset_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.image_info = []
        image_names = next(os.walk(dataset_dir))[2]
        for image_name in image_names:
            self.image_info.append(
                {'image_id': image_name, 'image_path': os.path.join(dataset_dir, image_name)})

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        image_info = self.image_info[index]
        image = cv2.imread(image_info['image_path'], 0)
        if self.transform:
            image, _ = self.transform(image)
        return image_info['image_id'], image


def test(data_dir, cfg,  image_size=(1600, 256)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     device = torch.device('cpu')
    WEIGHT_PATH = 'weights'
    MODEL_NAME = os.path.join(WEIGHT_PATH, 'unet_plus_99.pth')

    # 加载模型
    model = Unet_plus(1, 4).to(device)

    model.load_state_dict(torch.load(MODEL_NAME))

    model.eval()

    eval_dataset = EvalImageDataset(
        data_dir, transform=ImageTransform(image_size=image_size, mean=cfg.mean, std=cfg.std))

    eval_loader = data.DataLoader(eval_dataset, batch_size=2,
                                  shuffle=False, num_workers=0, collate_fn=detection_collate, pin_memory=True)

    submission = []
    with torch.no_grad():
        for image_path, image in eval_loader:
            image = image.to(device)
            output = model(image)
            pred = output.cpu().numpy()
            pred_mask,num=postMask(pred, cfg.threshold, cfg.min_size)
            if output is not None:
                for i, masks in enumerate(pred_mask):
                    for j, mask in enumerate(masks):
                        str_run_length = mask2rle(mask)
                        image_id = image_path[i]+'_'+str(j+1)
                        submission.append([image_id, str_run_length])

    # Save to csv file
    df = pd.DataFrame(submission, columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    DATA_DIR = '/home/guijiyang/dataset/severstal_steel/test_images'
    cfg = detectConfig()
    test(DATA_DIR, cfg)
