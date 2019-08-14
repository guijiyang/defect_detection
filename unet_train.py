from tqdm import tqdm
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch
from ImgDataset import ImageDataset
from transform import ImageTransform
from utils import computeDice, adjustStepLR, detectionCollate
from unet import UNet
from unetplus import Unet_plus
from loss import FocalLoss, DceDiceLoss
from config import detectConfig
from logger import Logger
from detect.unetplus import UnetPlus

import os
path = os.path.abspath(__file__)
os.chdir(os.path.dirname(path))
# import torchsummary
from thop import profile
# from sklearn.cross_validation import train_test_split


def train(data_dir, cfg, restart_train, epoch=1):
    logger = Logger('log', 'defect_detection')
    logger('开始训练')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    WEIGHT_PATH = 'weights'
    MODEL_NAME = os.path.join(WEIGHT_PATH, 'unet_plus.pth')
    metal_dataset = ImageDataset(
        data_dir, transform=ImageTransform(image_size=cfg.image_size, mean=cfg.mean, std=cfg.std))
    dataset_lens = len(metal_dataset)
    train_data_lens = int(dataset_lens*cfg.data_split)
    eval_data_lens = dataset_lens-train_data_lens
    batch_split = (train_data_lens//cfg.batch_size)//5
    logger('train : {}, eval : {}, batch_split : {}'.format(
        train_data_lens, eval_data_lens, batch_split))

    # 加载模型
    # model = UNet(image_size=min(cfg.image_size)).to(device)
    # model = Unet_plus(1, 4, mode='train').to(device)
    model=UnetPlus('resnet18',classes=4,inference_layer=4).to(device)
    loss_network = FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha, reduction='mean').to(device)
    # loss_network = DceDiceLoss(alpha=0.5, beta=1.).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    lr = adjustStepLR(optimizer, epoch, cfg.adjust_iter,
                      cfg.learning_rate, decay=cfg.lr_decay)
    # 统计模型参数
    # torchsummary.summary(model, (1,512,512),device='cpu')
    # 统计模型FLOPS
    # input_flops=torch.randn(1,3,1600,256).to(device)
    # flops,params=profile(model,inputs=( input_flops,))
    # logger('模型的FlOPS : {}, 参数量 : {}'.format(flops,params))

    if restart_train == True:
        if not os.path.exists(WEIGHT_PATH):
            os.mkdir(WEIGHT_PATH)
    else:
        # 加载前面训练得到模型权重
        if epoch > 0:
            if os.path.exists(MODEL_NAME):
                model.load_state_dict(torch.load(MODEL_NAME))
            else:
                raise Exception('cannot find model weights')
    logger('起始epoch：{}'.format(epoch))

    prev_dice_ious = 0
    while epoch <= cfg.max_epochs:
        train_data, eval_data = random_split(
            metal_dataset, [train_data_lens, eval_data_lens])

        train_loader = data.DataLoader(train_data, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=0, collate_fn=detectionCollate, pin_memory=True)
        eval_loader = data.DataLoader(eval_data, batch_size=cfg.batch_size,
                                      shuffle=True, num_workers=0, collate_fn=detectionCollate, pin_memory=True)
        model.train()
        losses, bce_losses, dice_losses = 0, 0, 0
        for idx, (_, images, target) in enumerate(train_loader):

            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss, bce_loss, dice_loss = loss_network(output.sigmoid_(), target)
            losses += loss
            bce_losses += bce_loss
            dice_losses += dice_loss
            loss.backward()
            optimizer.step()
            if idx % batch_split == 0 and idx != 0:
                logger(" batchs : {}, lr : {} |  loss :  {:.6f}, bce : {:.6f}, dice : {:.6f} | current loss : {:.6f}, bce : {:6f}, dice : {:.6f}".format(
                    idx, lr, losses/batch_split, bce_losses /
                    batch_split, dice_losses/batch_split,
                    loss, bce_loss, dice_loss))
                losses, bce_losses, dice_losses = 0, 0, 0

        model.eval()
        dice_ious = 0
        with torch.no_grad():
            for _, images, target in eval_loader:
                images, target = images.to(device), target.to(device)
                output = model(images)
                output = torch.where(
                    output > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.).to(device))
                dice_iou = computeDice(output, target, reduction='sum')
                dice_ious += float(dice_iou)
        dice_ious /= eval_data_lens
        # save whole model if current iou> prev iou
        if dice_ious > prev_dice_ious:
            torch.save(model.state_dict(), MODEL_NAME)
            prev_dice_ious = dice_ious
        logger("epoch : {}, dice_iou : {}".format(epoch, dice_ious))
        lr = adjustStepLR(optimizer, epoch, cfg.adjust_iter,
                          cfg.learning_rate, decay=cfg.lr_decay)
        epoch += 1


if __name__ == "__main__":
    data_dir = '/home/guijiyang/dataset/severstal_steel'
    # 训练
    cfg = detectConfig()
    cfg.batch_size = 1
    cfg.image_size = (1600, 256)  # W,H
    cfg.display()
    train(data_dir, cfg, True)
