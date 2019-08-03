import torch.nn as nn

def conv_Relu(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class CompactNet(nn.Module):
    r"""
    A compact convolution neural network use to classify metallic surface defect
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.layers=nn.Sequential(
            conv_Relu(227,55,11,4,0),   #1*227*227 => 96*55*55
            nn.MaxPool2d(3,2),   #96*55*55 => 96*27*27
            conv_Relu(96,129,5,1,0),    #96*27*27 => 128*23*23
            nn.MaxPool2d(3,2),  #128*23*23 => 128*11*11
            conv_Relu(128,256,3),  #128*11*11 => 256*11*11
            conv_Relu(256,256,3),  #256*11*11 => 256*11*11
            conv_Relu(256,128,3),  #256*11*11 => 128*11*11
            nn.MaxPool2d(3,2),  #128*11*11 => 128*5*5
            nn.Linear(128,1000),
            nn.Linear(1000,256),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        return self.layers(x)

