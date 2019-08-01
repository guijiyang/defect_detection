import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.alpha=alpha
        # if isinstance(alpha,(float,int)):
        #     self.alpha=torch.Tensor([alpha, 1-alpha])
        # if isinstance(alpha, list):
        #     self.alpha=torch.Tensor(alpha)
        self.size_average=size_average

    def forward(self, input, target):
        # if input.dim()>2:
        #     input=input.view(input.shape[0],input.shape[1],-1) #N,C,H,W => N,C,H*W
        #     input=input.transpose(1,2) #N,C,H*W => N,H*W,C
        #     input=input.contiguous().view(-1,input.shape[2]) #N,H*W,C => N*H*W,C
        # target=target.view(-1,1)
        pt=torch.where(target==1, input, 1-input)
        log_pt=torch.log(pt)
        alpha=torch.where(target==1, alpha,1-alpha)
        loss=-1*alpha*(1-pt)**self.gamma*log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()/input.shape[0]
