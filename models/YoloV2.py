# 此文件负责“调用骨干网络”并“加装检测头”
from torch import nn
from models.darknet import Darknet19


class YOLOv2(nn.Module):
    def __init__(self,num_classes,num_anchors):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # 骨干网络
        self.backbone = Darknet19() # out_channel:1024

        # 检测头 1*1 卷积
        self.head = nn.Conv2d(1024,num_anchors * (5 + num_classes),kernel_size=1)

    def forward(self,x):
        # 提取特征 (B, 1024, 13, 13)
        feature = self.backbone(x)

        # 输出结果 (B, num_anchors * (5 + num_classes), 13, 13) --> 此模型结果应为(B,45,13,13)
        output = self.head(feature)

        # 维度变换: (B, 45, 13, 13) -> (B, 13, 13, 45) -> (B, 13, 13, 5, 9)
        # contiguous: 返回一个连续的内存张量 ， 包含与自张量相同的数据。如果 自张量已经处于指定的内存格式中，该函数返回 自张量。
        output = output.permute(0, 2, 3, 1).contiguous() # 加上 contiguous 是个好习惯
        # .size(dim=0): 返回通道数
        output = output.view(output.size(dim=0), 13, 13, self.num_anchors, 5 + self.num_classes)

        return output