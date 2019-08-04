import torch.nn as nn


def conv_Relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bnormal=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels) if use_bnormal else nn.Sequential(),
        nn.ReLU()
    )


class CompactNet(nn.Module):
    r"""
    A compact convolution neural network use to classify metallic surface defect
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.layers = nn.Sequential(
            conv_Relu(1, 96, 11, 4, 0),  # 1*227*227 => 96*55*55
            nn.MaxPool2d(3, 2),  # 96*55*55 => 96*27*27
            conv_Relu(96, 128, 5, 1, 0),  # 96*27*27 => 128*23*23
            nn.MaxPool2d(3, 2),  # 128*23*23 => 128*11*11
            # 128*11*11 => 256*11*11
            conv_Relu(128, 256, 3, use_bnormal=False),
            # 256*11*11 => 256*11*11
            conv_Relu(256, 256, 3, use_bnormal=False),
            # 256*11*11 => 128*11*11
            conv_Relu(256, 128, 3, use_bnormal=False),
            nn.MaxPool2d(3, 2),  # 128*11*11 => 128*5*5
        )
        self.fcs = nn.Sequential(
            nn.Linear(128*5*5, 1000),
            nn.Linear(1000, 256),
            nn.Linear(256, num_classes)
        )

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
        in_size = x.shape[0]
        output = self.layers(x)
        output = output.view(in_size, -1)
        return self.fcs(output)
