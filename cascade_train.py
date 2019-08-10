from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch
from ImgDataset import ImageDataset
from transform import ImageTransform
from utils import  computeDice
from unet import UNet
from unetplus import Unet_plus
from cascade import CascadeNet
from loss import FocalLoss, DceDiceLoss
from config import detectConfig
from logger import Logger

import os
path = os.path.abspath(__file__)
os.chdir(os.path.dirname(path))
# import torchsummary
# from thop import profile

# from sklearn.cross_validation import train_test_split


def detection_collate(batch):
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
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return torch.stack(imgs, 0), torch.stack(targets, 0)


def train(data_dir, cfg, restart_train, epoch=1):
    logger = Logger('log', 'defect_detection')
    logger('开始训练')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    WEIGHT_PATH = 'weights'
    MODEL_FIRST = os.path.join(WEIGHT_PATH, 'unet_first_60.pth')
    MODEL_SECOND = os.path.join(WEIGHT_PATH, 'unet_second_{}.pth')
    metal_dataset = ImageDataset(
        data_dir, transform=ImageTransform(image_size=cfg.image_size))
    dataset_lens = len(metal_dataset)
    train_data_lens = int(dataset_lens*0.9)
    eval_data_lens = dataset_lens-train_data_lens
    batch_split = (train_data_lens//cfg.batch_size)//5
    logger('train : {}, eval : {}, batch_split : {}'.format(
        train_data_lens, eval_data_lens, batch_split))

    # 加载模型
    network_1 = Unet_plus(1, 1)
    network_2 = Unet_plus(1, 1)
    model = CascadeNet(network_1, network_2).to(device)
    model.train_only_for_2()
    # model = Unet_plus(1, 1, mode='train').to(device)
    # loss_network = FocalLoss(gamma=0., alpha=0.8, size_average=False).to(device)
    loss_network = DceDiceLoss(alpha=0.5, beta=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.adjust_iter, gamma=cfg.lr_decay, last_epoch=-1)

    # 统计模型参数
    # torchsummary.summary(model, (1,512,512),device='cpu')
    # 统计模型FLOPS
    # input_flops=torch.randn(1,1,512,512)
    # flops,params=profile(model,inputs=( input_flops,))
    # logger('模型的FlOPS : {}, 参数量 : {}'.format(flops,params))

    if restart_train == True:
        if os.path.exists(MODEL_FIRST):
            model.load_state_dict(torch.load(MODEL_FIRST))
        else:
            raise Exception('cannot find model weights')
    else:
        # 加载前面训练得到模型权重
        if epoch > 0:
            if os.path.exists(MODEL_SECOND.format(epoch-1)):
                # model.load_state_dict(torch.load(MODEL_SECOND.format(epoch-1)))
                model.load_state_dict(torch.load(MODEL_FIRST), torch.load(
                    MODEL_SECOND.format(epoch-1)))
            else:
                raise Exception('cannot find model weights')
    logger('起始epoch：{}'.format(epoch))
    while epoch <= cfg.max_epochs:
        train_data, eval_data = random_split(
            metal_dataset, [train_data_lens, eval_data_lens])

        train_loader = data.DataLoader(train_data, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=0, collate_fn=detection_collate, pin_memory=True)
        eval_loader = data.DataLoader(eval_data, batch_size=cfg.batch_size,
                                      shuffle=True, num_workers=0, collate_fn=detection_collate, pin_memory=True)
        model.train()
        losses = 0
        for idx, (images, target) in enumerate(train_loader):

            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_network(output[-1], target)
            losses += loss.data
            loss.backward()
            optimizer.step()
            if idx % batch_split == 0 and idx != 0:
                logger("epoch : {}, batchs : {}, loss :  {:.6f}".format(
                    epoch, idx, losses/(cfg.batch_size*batch_split)))
                losses = 0
        # save whole model
        torch.save(model.get_state_dict_net2(), MODEL_SECOND.format(epoch))

        model.eval()
        dice_ious = 0
        with torch.no_grad():
            for images, target in eval_loader:
                images, target = images.to(device), target.to(device)
                output = model(images)
                output = torch.where(
                    output[-1] > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.).to(device))
                dice_iou = computeDice(output, target, reduction='sum')
                dice_ious += float(dice_iou)
        dice_ious /= eval_data_lens
        logger("epoch : {}, dice_iou : {}".format(epoch, dice_ious))
        # scheduler.step()
        epoch += 1


def test(data_dir, image_size=(1600, 256)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     device = torch.device('cpu')
    WEIGHT_PATH = 'weights'
    MODEL_NAME = os.path.join(WEIGHT_PATH, 'unet_first.pth')

    # 加载模型
    model = Unet_plus(1, 1).to(device)

    model.load_state_dict(torch.load(MODEL_NAME))

    model.eval()

    eval_dataset = ImageDataset(
        data_dir, transform=ImageTransform(image_size=image_size))

    eval_loader = data.DataLoader(eval_dataset, batch_size=16,
                                  shuffle=True, num_workers=4, collate_fn=detection_collate, pin_memory=True)

    dice_ious = 0
    for images, target in eval_loader:
        images, target = images.to(device), target.to(device)
        output = model(images)
        output = torch.where(
            output[-1] > 0.5, torch.tensor(1.0).to(device), torch.tensor(0.).to(device))
        dice_iou = computeDice(output, target, reduction='sum')
        dice_ious += float(dice_iou)
    print(dice_ious/len(eval_dataset))


if __name__ == "__main__":
    data_dir = '/home/guijiyang/dataset/severstal_steel'
    # 训练
    cfg = detectConfig()
    cfg.batch_size = 2
    cfg.image_size = (1600, 256)  # W,H
    cfg.display()
    train(data_dir, cfg, True)
    # 测试
    # test(data_dir, image_size=(256,256))
