import torch
import colorsys
import random
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


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


# 显示带有mask的图像
def displayTopMasks(image, masks, class_ids=None):
    """Display the given image and the top few class masks."""
    colors = random_colors(masks.shape[0])
    img = image.copy()
    for idx in range(masks.shape[0]):
        img = applyMask(img, masks[idx], color=colors[idx])
    fig = plt.figure(figsize=(8, 16))
    plt.imshow(img)
    plt.title('class id: {}'.format(class_ids))
    plt.xticks([])
    plt.yticks([])

    plt.show()


def applyMask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(image.shape[-1]):
        image[:, :, c] = np.where(mask[:, :, c] == 1.0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def computeDice(pred_mask, gt_mask, p=1, epilson=1e-6, reduction='mean'):
    r"""Computes IoU overlaps between two sets of masks.

   gt_masks, pred_masks: [batch, 1, Height, Width]
   \text{dice}=\frac{2*TP}{2TP+FP+FN}
   """
    pred_mask = pred_mask.type(torch.float)
    gt_mask = gt_mask.type(torch.float)
    pred_mask_f = pred_mask.contiguous().view(pred_mask.shape[0], -1)
    gt_mask_f = gt_mask.contiguous().view(gt_mask.shape[0], -1)
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


def detectionCollate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    img_path = []
    targets = []
    imgs = []
    for sample in batch:
        img_path.append(sample[0])
        imgs.append(sample[1])
        targets.append(sample[2])
    return img_path, torch.stack(imgs, 0), torch.stack(targets, 0)
