import torch
import numpy as np
import matplotlib.pyplot as plt


def mask2rle(img):
    """ img : numpy array, 1 - mask, 0 - background
    Returns run length as string formated 
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]+1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


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
def displayTopMasks(image, masks, class_ids=None):
    """Display the given image and the top few class masks."""
    fig = plt.figure(figsize=(14, 14))
    for idx in range(len(masks)):
        img = image.copy()
        img = applyMask(img, masks[idx], color=[0.5, 0.5, 0.5])
        img = np.squeeze(img, axis=-1)
        ax = plt.subplot(2, 2, idx+1)
        ax.imshow(img, cmap='gray')
        ax.set_title('class id: {}'.format(idx))
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def applyMask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(image.shape[-1]):
        image[:, :, c] = np.where(mask[:, :, c] == 1.0,
                                  0,
                                  image[:, :, c])
    return image


def computeDice(pred_mask, gt_mask, p=1, epilson=1e-6, reduction='mean'):
    r"""Computes IoU overlaps between two sets of masks.

   gt_masks, pred_masks: [batch, 1, Height, Width]
   \text{dice}=\frac{2*TP}{2TP+FP+FN}
   """
    pred_mask_f = pred_mask.contiguous().view(pred_mask.shape[0], -1)
    gt_mask_f = gt_mask.contiguous().view(gt_mask.shape[0], -1)
    gt_mask_f = gt_mask_f.type_as(pred_mask_f)
    tp = torch.sum(torch.mul(gt_mask_f, pred_mask_f), dim=1)
    den = torch.sum(pred_mask_f.pow(p)+gt_mask_f.pow(p), dim=1)
    dice = (2*tp+epilson)/(den+epilson)
    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    elif reduction == 'none':
        return dice
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))
# def dice_loss()


def adjustStepLR(optimizer, epoch, adjust_lr_epoch=10, init_lr=1e-3, decay=0.8, min_lr=1e-6):
    """
    Learning rate decay with epoch changed
    """
    lr_change_num = epoch//adjust_lr_epoch
    lr = max(init_lr * (decay ** (lr_change_num)), min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
