import numpy as np
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma=gamma
        self.alpha=torch.tensor(alpha)
        # if isinstance(alpha,(float,int)):
        #     self.alpha=torch.Tensor([alpha, 1-alpha])
        # if isinstance(alpha, list):
        #     self.alpha=torch.Tensor(alpha)
        self.size_average=size_average
    def to(self,device):
        self.alpha=self.alpha.to(device)
        return super().to(device)

    def forward(self, pred, target):
        # if input.dim()>2:
        #     input=input.view(input.shape[0],input.shape[1],-1) #N,C,H,W => N,C,H*W
        #     input=input.transpose(1,2) #N,C,H*W => N,H*W,C
        #     input=input.contiguous().view(-1,input.shape[2]) #N,H*W,C => N*H*W,C
        # target=target.view(-1,1)
        pt=torch.where(target==1, pred, 1-pred)
        log_pt=torch.log(pt)
        device=pred.device
        # alpha_t=torch.as_tensor(self.alpha)
        alpha=torch.where(target==1,self.alpha ,1-self.alpha)
        loss=-1*alpha*(1-pt)**self.gamma*log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()/pred.shape[0]

a=np.random.rand(5,1,512,512)
# print(a)
b=np.random.randint(0,2,(5,1,512,512))
# print(b)
alpha=torch.tensor(0.3)
gamma=1
a_t=torch.as_tensor(a,dtype=torch.float32)
b_t=torch.as_tensor(b)

device=torch.device('cuda')

a_t=a_t.to(device)
b_t=b_t.to(device)
loss_net=FocalLoss(gamma=1,alpha=0.5, size_average=False).to(device)
loss=loss_net(a_t, b_t)