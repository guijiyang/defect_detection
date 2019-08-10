import torch.nn as nn

# from unet import UNet
# from unetplus import Unet_plus


class CascadeNet(nn.Module):
    def __init__(self, networks_1,networks_2):
        super(CascadeNet, self).__init__()
        self.unet_1 = networks_1
        self.unet_2 = networks_2

    def load_state_dict(self, state_dict_1, state_dict_2=None, strict=True):
        self.unet_1.load_state_dict(state_dict_1)
        if state_dict_2 is not None:
            self.unet_2.load_state_dict(state_dict_2)

    def get_state_dict_net2(self):
        return self.unet_2.state_dict()

    def train_only_for_2(self):
        for param in self.unet_1.parameters():
            param.requires_grad=False

    def forward(self, x):
        output = self.unet_1(x)
        return self.unet_2(output[-1])
