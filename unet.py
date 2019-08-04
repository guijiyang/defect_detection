import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_3X3_Relu(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def atrous_conv_3X3_Relu(in_channels, out_channels, stride=1, dilation=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride,
                  padding=2, dilation=2,  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class down_sample_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_sample_conv, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            atrous_conv_3X3_Relu(in_channels, out_channels),
            conv_3X3_Relu(out_channels, out_channels))

    def forward(self, x):
        return self.layer(x)


class up_sample_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sample_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.layer = nn.Sequential(
            conv_3X3_Relu(in_channels, out_channels),
            conv_3X3_Relu(out_channels, out_channels))

    def forward(self, x, y):
        x = self.conv(x)
        output = torch.cat([x, y], dim=1)
        output = self.layer(output)
        return F.interpolate(output, scale_factor=2, mode='bilinear')


class UNet(nn.Module):
    """ UNet network for semantic  segmentation """

    def __init__(self, image_size=(512, 512)):
        super(UNet, self).__init__()
        self.image_size = image_size
        self.start = nn.Sequential(
            conv_3X3_Relu(1, 64),
            conv_3X3_Relu(64, 64))
        # down sample layers
        self.dp1 = down_sample_conv(64, 128)
        self.dp2 = down_sample_conv(128, 256)
        if self.image_size[0] == 512:
            self.dp3 = down_sample_conv(256, 512)

            self.dp4 = down_sample_conv(512, 1024)

            # up sample layers
            self.up3 = up_sample_conv(1024, 512)
        else:  # image_size[0]=256
            self.dp3 = down_sample_conv(256, 512)
        self.up2 = up_sample_conv(512, 256)
        self.up1 = up_sample_conv(256, 128)

        self.conv = nn.Conv2d(128, 64, 1, bias=False)
        self.end = nn.Sequential(
            conv_3X3_Relu(128, 64),
            nn.Conv2d(64, 1, kernel_size=1, bias=False))
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        concats = []
        firstOut = self.start(x)             # 64*512*512 or 64*256*256
        down1 = self.dp1(firstOut)  # 128*256*256 or 128*128*128
        down2 = self.dp2(down1)  # 256*128*128 or 256*64*64
        if self.image_size[0] == 512:
            down3 = self.dp3(down2)  # 512*64*64
            down4 = self.dp4(down3)  # 1024*32*32
            up4 = F.interpolate(down4, scale_factor=2,
                                mode='bilinear')  # 1024*64*64
            up3 = self.up3(up4, down3)  # 512*128*128
            up2 = self.up2(up3, down2)  # 256*256*256
        else:
            down3 = self.dp3(down2)  # 512*32*32
            up3 = F.interpolate(down3, scale_factor=2,
                                mode='bilinear')  # 512*64*64
            up2 = self.up2(up3, down2)  # 256*128*128
        up1 = self.up1(up2, down1)  # 128*512*512 or 128*256*256
        convOut = self.conv(up1)  # 64*512*512 or 64*256*256
        # 128*512*512 or 128*256*256
        output = torch.cat([convOut, firstOut], dim=1)
        output = self.end(output)  # 1*512*512 or 1*256*256
        return torch.sigmoid(output)
        # return output
