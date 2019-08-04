from torch.utils.data.dataset import random_split
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from thop import profile
from ImgDataset import MaskDataset
from classifier import CompactNet
from logger import Logger
from config import ClassifyConfig
from transform import MaskTransform
import torch
import os
path = os.path.abspath(__file__)
os.chdir(os.path.dirname(path))
# from torchvision import transforms

data_dir = '/home/guijiyang/dataset/severstal_steel'


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
    return torch.stack(imgs, 0), torch.tensor(targets)


def train(restart_train, data_dir,  cfg):
    logger = Logger('log', 'classifier')
    logger('开始训练')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device=torch.device('cpu')
    WEIGHT_PATH = 'weights'
    GLOBAL_STEP_FILE = os.path.join(WEIGHT_PATH, 'epoch.log')
    MODEL_NAME = os.path.join(WEIGHT_PATH, 'compactNet_{}.pth')
    mask_dataset = MaskDataset(data_dir, transform=MaskTransform(image_size=cfg.image_size))

    dataset_lens = len(mask_dataset)
    train_data_lens = int(dataset_lens*0.7)
    eval_data_lens = dataset_lens-train_data_lens
    logger('train : {}, eval : {}'.format(train_data_lens, eval_data_lens))

    # 加载模型
    model = CompactNet(cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate,
                          momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.adjust_iter, gamma=cfg.lr_decay, last_epoch=-1)

    # # 统计模型FLOPS
    # input_flops=torch.randn(1,1,227,227).to(device)
    # flops,params=profile(model,inputs=( input_flops,))
    # logger('模型的FlOPS : {}, 参数量 : {}'.format(flops,params))

    if restart_train == True:
        if not os.path.exists(WEIGHT_PATH):
            os.mkdir(WEIGHT_PATH)
        epoch = 0
    else:
        if os.path.exists(GLOBAL_STEP_FILE):
            with open(GLOBAL_STEP_FILE, 'r') as f:
                epoch = int(f.read())
        else:
            raise Exception('cannot find global step file')
        # 加载模型权重
        if epoch > 0:
            if os.path.exists(MODEL_NAME.format(epoch)):
                model.load_state_dict(torch.load(MODEL_NAME.format(epoch)))
            else:
                raise Exception('cannot find model weights')
    logger('起始epoch：{}'.format(epoch))
    losses = 0
    loss_epochs=[]
    while epoch <= cfg.max_epochs:
        train_data, eval_data = random_split(
            mask_dataset, [train_data_lens, eval_data_lens])

        train_loader = data.DataLoader(train_data, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=4, collate_fn=detection_collate, pin_memory=True)
        eval_loader = data.DataLoader(eval_data, batch_size=cfg.batch_size,
                                      shuffle=True, num_workers=4, collate_fn=detection_collate, pin_memory=True)
        model.train()
        loss_epoch=0
        for idx, (images, target) in enumerate(train_loader):

            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, target)
            losses += loss.data
            loss.backward()
            optimizer.step()
            if idx %10 == 0 and idx !=0:
                print('.', end='')
            if idx % 100 == 0 and idx != 0:
                logger("epoch : {}, batchs : {}, loss :  {:.6f}".format(
                    epoch, idx, losses/100))
                loss_epoch+=losses
                losses = 0
        # save whole model
        torch.save(model.state_dict(), MODEL_NAME.format(epoch))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, target in eval_loader:
                images, target = images.to(device), target.to(device)
                output = model(images)
                pred = torch.argmax(output, dim=1)
                correct += (pred == target).sum().float()
                total += target.shape[0]
        logger("epoch : {}, accuracy : {}".format(epoch, correct/total))
        
        # 损失停滞，则终止训练
        if len(loss_epochs)>1 and loss_epoch>=0.98*loss_epochs[-1] and loss_epoch>=0.98*loss_epochs[-2]:
            logger('损失停滞，中断训练，当前损失：{} ，前面损失：{}'.format(loss_epoch, loss_epochs[-1]))
            break
        loss_epochs.append(loss_epoch)
        epoch += 1

if __name__ == "__main__":
    cfg = ClassifyConfig()
    cfg.display()
    train(True, data_dir, cfg)
