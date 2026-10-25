import torch
from models.YoloV2 import YOLOv2

if __name__=='__main__':
    model = YOLOv2(num_classes=4,num_anchors=5)
    # 模拟一个 batch 的数据
    dummy_input = torch.randn(2,3,416,416)

    # 前向传播
    try:
        output = model(dummy_input)
        print("网格测试成功")
        print("输入形状",dummy_input.shape)
        print("输出形状",output.shape)
    except Exception as e:
        print("失败：\n",e)