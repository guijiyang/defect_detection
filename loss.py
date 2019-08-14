import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import computeDice


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, epsilon=1e-6, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = torch.as_tensor(alpha)
        self.epsilon = torch.as_tensor(epsilon)
        self.reduction = reduction

    def to(self, device):
        self.alpha = self.alpha.to(device)
        super().to(device)
        return self

    def forward(self, pred, target):
        pt = torch.where(target == 1, pred+self.epsilon, 1-pred+self.epsilon)
        log_pt = torch.log(pt)
        alpha = torch.where(target == 1, self.alpha, 1-self.alpha)
        loss = -1*alpha*(1-pt)**self.gamma*log_pt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class DceDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1., reduction='mean'):
        super(DceDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(
            pred, target.type_as(pred), reduce=self.reduction)
        dice_loss = 1.-computeDice(pred, target, reduction=self.reduction)
        return self.alpha*bce_loss+self.beta*dice_loss, bce_loss, dice_loss