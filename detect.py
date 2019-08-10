from unetplus import Unet_plus
from classifier import CompactNet

import cv2
import math
import torch
import numpy as np
import torch.nn as nn

class DetectNet(nn.Module):
    def __init__(self, threshold, min_size, cascade_size=512, classifier_size=(227,227), num_classes=4):
        super(DetectNet, self).__init__()
        self.threshold=threshold
        self.min_size=min_size
        self.cascade_size=cascade_size
        self.classifier_size=classifier_size
        self.num_classes=num_classes
        self.cascade=Unet_plus(1,1,mode='test')
        self.classifier=CompactNet(num_classes=num_classes)

    def load_state_dict(self, cascade_dict, classifier_dict, strict=True):
        self.cascade.load_state_dict(cascade_dict)
        self.classifier.load_state_dict(classifier_dict)


    def forward(self, x):
        image_shape=x.shape
        # 预测mask
        pred_mask_t = self.cascade(x)
        device=pred_mask_t.device
        image=x.reshape(image_shape[-2:]).cpu().clone().numpy()
        pred_mask=pred_mask_t.cpu().clone().numpy()
        pred_mask=pred_mask.reshape(image_shape[-2:])
        mask_t=cv2.threshold(pred_mask,self.threshold,1, cv2.THRESH_BINARY)[1]
        # 然后计算mask中的所有连通区域，得到的不同的mask对应的值不同，这样就可以进一步分离每个mask
        num_component,component=cv2.connectedComponents(mask_t.astype(np.uint8))
        if num_component<1:
            return None

        masks=np.zeros((self.num_classes,*image_shape[-2:]), dtype=np.uint8)
        for i in range(1,num_component):
            points=(component==i)
            if points.sum() < self.min_size:
                continue
            points_value=np.where(points)
            points_value=np.stack([points_value[1],points_value[0]], axis=1)
            # 求mask的最小包围矩形，得到的是矩形的左上角点，长宽，和水平x轴的逆向夹角
            rect=cv2.minAreaRect(points_value)
            width=rect[1][0]
            height=rect[1][1]
            # 如果矩形面积等于0，则跳过分割mask图形过程
            if width <=0. or height<=0.:
                    continue
                
            # 将rect转化为矩形的四个角点
            box=cv2.boxPoints(rect)
            # box=np.int64(box)
            src_box=box.astype(np.float32)
            # 目标矩形
            dst_box=np.array([[0, height-1],[0,0],[width-1,0],[width-1,height-1]], dtype=np.float32)
            # 建立源矩形到目标矩形的映射矩阵
            M=cv2.getPerspectiveTransform(src_box,dst_box)
            warp=cv2.warpPerspective(image, M, (math.ceil(width), math.ceil(height)))
            warp=cv2.resize(warp, self.classifier_size, interpolation=cv2.INTER_LINEAR)
            classifier_input=torch.as_tensor(warp.reshape(1,1,*self.classifier_size)).to(device)
            # 预测mask分类
            classifier_pred= self.classifier(classifier_input)

            pred_id = torch.argmax(classifier_pred, dim=1)
            
            masks[pred_id[0],points_value[:,1],points_value[:,0]]=1

        return masks