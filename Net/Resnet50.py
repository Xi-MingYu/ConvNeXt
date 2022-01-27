import torch
from torch import nn

class BottleNeck (nn.Module):

    extention = 4 #每个stage中维度拓展倍数
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BottleNeck, self).__init__()

        self.conv1 =nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=stride,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels*self.extention,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward (self, x):
        residual = x   #残差操作
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        if self.downsample != None:
            residual = self.downsample(x)
        out += residual   #残差操作
        out = self.relu(out)

        return out
class ResNet50(nn.Module):
    def __init__(self, block, layers):
        super(ResNet50,self).__init__()

        self.in_channels = 64
        self.block = block
        self.layers = layers

        self.conv1=nn.Conv2d(in_channels=3,out_channels=self.in_channels,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn=nn.BatchNorm2d(self.in_channels)
