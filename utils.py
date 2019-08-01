import torch
import numpy as np
import matplotlib.pyplot as plt


def rle2mask(rle, imgshape):
    width = imgshape[1]
    height = imgshape[0]

    mask = np.zeros(width*height).astype(np.uint8)
    if rle != None:
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            mask[int(start):int(start+lengths[index])] = 1
            current_position += lengths[index]

    return np.transpose(mask.reshape(width, height))


# 显示带有mask的图像
def display_top_masks(image, masks, class_ids=None):
    """Display the given image and the top few class masks."""
    fig = plt.figure(figsize=(14, 14))
    for idx in range(len(masks)):
        img = image.copy()
        img = apply_mask(img, masks[idx], color=[0.5, 0.5, 0.5])
        img = np.squeeze(img, axis=-1)
        ax = plt.subplot(2, 2, idx+1)
        ax.imshow(img, cmap='gray')
        ax.set_title('class id: {}'.format(idx))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(image.shape[-1]):
        image[:, :, c] = np.where(mask[:, :, c] == 1.0,
                                  0,
                                  image[:, :, c])
    return image


def compute_dice(gt_mask, pred_mask, alpha=0.5, beta=0.5):
    r"""Computes IoU overlaps between two sets of masks.

   gt_masks, pred_masks: [batch, 1, Height, Width]
   \text{dice}=\frac{TP}{2TP+FP+FN}
   """
    gt_mask_f = gt_mask.flatten()
    pred_mask_f = pred_mask.flatten()
    pred_mask_f=pred_mask_f.type_as(gt_mask_f)
    TP = torch.sum(gt_mask_f*pred_mask_f)
    gt_sum=torch.sum(gt_mask_f)
    pred_sum=torch.sum(pred_mask_f)
    if gt_sum == 0 and pred_sum == 0:
        return torch.tensor(1.0)
    return (2*TP)/(gt_sum+pred_sum)