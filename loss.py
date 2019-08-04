import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.alpha=torch.as_tensor(alpha)
        # if isinstance(alpha,(float,int)):
        #     self.alpha=torch.Tensor([alpha, 1-alpha])
        # if isinstance(alpha, list):
        #     self.alpha=torch.Tensor(alpha)
        self.size_average=size_average
    
    def to(self,device):
        self.alpha=self.alpha.to(device)
        super().to(device)
        return self

    def forward(self, pred, target):
        # if input.dim()>2:
        #     input=input.view(input.shape[0],input.shape[1],-1) #N,C,H,W => N,C,H*W
        #     input=input.transpose(1,2) #N,C,H*W => N,H*W,C
        #     input=input.contiguous().view(-1,input.shape[2]) #N,H*W,C => N*H*W,C
        # target=target.view(-1,1)
        pt=torch.where(target==1, pred, 1-pred)
        log_pt=torch.log(pt)
        alpha=torch.where(target==1, self.alpha,1-self.alpha)
        loss=-1*alpha*(1-pt)**self.gamma*log_pt
        if self.size_average:
            return loss.mean()
        else:
            batch_loss = loss.sum(dim=(1,2,3))
            return batch_loss.mean()
