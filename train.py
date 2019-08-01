from ImgDataset import ImageDataset
from transform import ImageTransform
from utils import display_top_masks,compute_dice
from unet import UNet
from loss import FocalLoss
from config import Config
from logger import Logger

import os
path=os.path.abspath(__file__)
os.chdir(os.path.dirname(path))
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
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
    return torch.stack(imgs, 0), torch.stack(targets,0)


def train(restart_train, data_dir, cfg):
    logger = Logger('log', 'defect_detection')
    logger('开始训练')
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device=torch.device('cpu')
    WEIGHT_PATH='weights'
    GLOBAL_STEP_FILE=os.path.join(WEIGHT_PATH,'epoch.log')
    MODEL_PTH=os.path.join(WEIGHT_PATH,'unet_1.pth')
    train_dataset = ImageDataset(
        data_dir, mode='train',split_ratio=0.9, transform=ImageTransform(image_size=(512, 512)))
    train_dataset_lens=len(train_dataset)
    logger('训练数据集 ：{}'.format(train_dataset_lens))
    test_dataset = ImageDataset(
        data_dir, mode='test',split_ratio=0.9, transform=ImageTransform(image_size=(512, 512)))
    logger('测试数据集 ：{}'.format(len(test_dataset)))
    # print(train_dataset[0])
    train_loader = data.DataLoader(train_dataset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=4, collate_fn=detection_collate, pin_memory=True)
    test_loader = data.DataLoader(test_dataset, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=4, collate_fn=detection_collate, pin_memory=True)
    
    #加载模型
    model=UNet(image_size=(512, 512)).to(device)
    loss_network=FocalLoss(gamma=1,alpha=0.5, size_average=False).to(device)
    optimizer= optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.adjust_iter, gamma=cfg.lr_decay, last_epoch=-1)

    # 统计模型参数
    # torchsummary.summary(model, (1,512,512),device='cpu')
    # 统计模型FLOPS
    # input_flops=torch.randn(1,1,512,512)
    # flops,params=profile(model,inputs=( input_flops,))
    # logger('模型的FlOPS : {}, 参数量 : {}'.format(flops,params))

    if restart_train ==True:
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
        if epoch>0:
            if os.path.exists(MODEL_PTH):
                model.load_state_dict(torch.load(MODEL_PTH))
            else:
                raise Exception('cannot find model weights')
    logger('起始epoch：{}'.format(epoch))
    losses = 0
    while epoch <= cfg.max_epochs:
        model.train()
        for idx,(images,target) in enumerate(train_loader):
            
            images,target=images.to(device), target.to(device)
            optimizer.zero_grad()
            output=model(images)
            loss=loss_network(output, target)
            losses+=loss.data
            loss.backward()
            optimizer.step()
            if idx%100==0 and idx!=0:
                logger("epoch : {}, relative_step : {}, loss :  {:.6f}".format(epoch,idx,losses/100))
                losses=0
            # save whole model
            torch.save(model.state_dict(), MODEL_PTH)
        
        model.eval()
        dice_ious=0
        with torch.no_grad():
            for images, target in test_loader:
                images,target=images.to(device), target.to(device)
                output=model(images)
                output=torch.where(output>0.5,torch.tensor(1.0),torch.tensor(0.))
                dice_iou=compute_dice(output, target)
                dice_ious+=dice_iou.data
        dice_ious/=(len(test_dataset))
        logger("epoch : {}, dice_iou : {}".format(epoch, dice_ious))
        scheduler.step()
        epoch+=1

if __name__ == "__main__":
    cfg = Config()
    cfg.display()
    train(True, data_dir='/home/guijiyang/dataset/severstal_steel', cfg)
