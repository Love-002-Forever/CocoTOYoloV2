# 此文件只负责提取特征
import torch
from torch import nn

class ConvBNLeakyReLU(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(ConvBNLeakyReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1,inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        self.features = nn.Sequential(
            # 第一阶段: 416x416x3 -> 208x208x32
            ConvBNLeakyReLU(3,32,3,1,1),
            nn.MaxPool2d(2,2),

            # 第二阶段: 208x208x32 -> 104x104x64
            ConvBNLeakyReLU(32,64,3,1,1),
            nn.MaxPool2d(2,2),

            # 第三阶段: 104x104x64 -> 52x52x128
            ConvBNLeakyReLU(64,128,3,1,1),
            ConvBNLeakyReLU(128,64,1,1,0),
            ConvBNLeakyReLU(64,128,3,1,1),
            nn.MaxPool2d(2,2),

            # 第四阶段: 52x52x128 -> 26x26x256
            ConvBNLeakyReLU(128,256,3,1,1),
            ConvBNLeakyReLU(256,128,1,1,0),
            ConvBNLeakyReLU(128,256,3,1,1),
            nn.MaxPool2d(2,2),

            # 第五阶段: 26x26x256 -> 13x13x512
            ConvBNLeakyReLU(256,512,3,1,1),
            ConvBNLeakyReLU(512,256,1,1,0),
            ConvBNLeakyReLU(256,512,3,1,1),
            ConvBNLeakyReLU(512,256,1,1,0),
            ConvBNLeakyReLU(256,512,3,1,1),
            nn.MaxPool2d(2,2),

            # 第六阶段: 13x13x512 -> 13x13x1024
            ConvBNLeakyReLU(512,1024,3,1,1),
            ConvBNLeakyReLU(1024,512,1,1,0),
            ConvBNLeakyReLU(512,1024,3,1,1),
            ConvBNLeakyReLU(1024,512,1,1,0),
            ConvBNLeakyReLU(512,1024,3,1,1),
            # 注意最后的输出通道必须是 1024！
        )

    def  forward(self,x):
        return self.features(x)

