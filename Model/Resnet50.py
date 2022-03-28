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
                              bias=False)#使用BN将bias参数置False
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

        self.conv1=nn.Conv2d(in_channels=3,
                             out_channels=self.in_channels,
                             kernel_size=7,
                             stride=2,
                             padding=3,
                             bias=False)
        #stem网络层
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
        #stage1-stage4由make_stage函数生成
        self.stage1 = self.make_stage(self.block,64,self.layers[0],stride=1)
        self.stage2 = self.make_stage(self.block,128,self.layers[1],stride=2)
        self.stage3 = self.make_stage(self.block,256,self.layers[2],stride=2)
        self.stage4 = self.make_stage(self.block,512,self.layers[3],stride=2)

    def forward (self, x):
        #stem部分:conv+bn+relu+maxpool
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        #block
        out=self.stage1(out)
        out=self.stage2(out)
        out=self.stage3(out)
        out=self.stage4(out)

        return out
    def make_stage(self, block, out_channels, block_num, stride=1):
        """
            block: block模块
            out_channels：每个模块中间运算的维度，一般等于输出维度/4
            block_num：重复次数
            stride：Conv Block的步长
        """
        block_list=[]
        downsample=None
        if (stride!=1 or self.in_channels!=out_channels*block.extention):
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*block.extention,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*block.extention)
            )
        #Conv_Block
        conv_block=block(self.in_channels,out_channels,stride=stride,downsample=downsample)
        block_list.append(conv_block)
        self.in_channels=out_channels*block.extention
        #Identity_Block
        for i in range (1,block_num):
            block_list.append(block(self.in_channels,out_channels,stride))

        return nn.Sequential(*block_list)



if __name__ == "__main__":
    resnet = ResNet50(BottleNeck, [3, 4, 6, 3])
    x = torch.randn(1, 3, 224, 224)
    # x=resnet(x)
    print(x.shape)

    # 导出onnx
    torch.onnx.export(resnet,x,'resnet50',verbose=True)
    pass