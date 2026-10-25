import torch
from torch import nn

class YOLOv2Loss(nn.Module):
    def __init__(self,cfg):
        super(YOLOv2Loss,self).__init__()
        self.cfg = cfg
        self.lambda_coord = cfg.lambda_coord
        self.lambda_noobj = cfg.lambda_noobj

    def forward(self,pred,target):
        """
        模型预测输出 torch.Size([B, 13, 13, 5, 9])
        真实标签    torch.Size([B, 13, 13, 5, 9])
        """
        B = pred.size(0)

        # 预处理预测值
        # 中心点 xy (0-1)
        pred_xy = torch.sigmoid(pred[...,0:2])
        # 宽高 wh (无需激活函数)
        pred_wh = pred[...,2:4]
        # 置信度 c (0-1)
        pred_conf = torch.sigmoid(pred[...,4:5])
        # 类别概率 (经过 softmax，注意 dim=-1)
        pred_cls = torch.softmax(pred[...,5:],dim=-1)

        # 提取真实标签
        target_xy = target[...,0:2]
        target_wh = target[...,2:4]
        target_conf = target[...,4:5]
        target_cls = target[...,5:]

        # 制作掩码 (Mask)
        obj_mask = target_conf == 1
        noobj_mask = target_conf == 0

        # 计算 4 个 Loss
        # 注意：利用 obj_mask 把没有物体的位置乘 0，只保留有物体的部分。
        # 提示：torch.sum() 可以求和，记得最后除以 batch_size (B) 来求平均
        # a. 中心点损失
        loss_xy = torch.sum((pred_xy - target_xy)**2 * obj_mask) / B
        # b. 宽高损失
        loss_wh = torch.sum((pred_wh - target_wh)**2 * obj_mask) / B
        # c. 置信度损失
        # 正样本 (有物体): 计算 pred_conf 和 1 的误差
        loss_conf_obj = torch.sum((pred_conf - target_conf)**2 * obj_mask) / B * self.cfg.lambda_obj
        # 负样本 (无物体): 计算 pred_conf 和 0 的误差，记得乘上 lambda_noobj
        loss_conf_noobj = torch.sum((pred_conf - target_conf)**2 * noobj_mask) / B * self.lambda_noobj
        # d. 类别损失
        loss_cls = torch.sum((pred_cls - target_cls)**2 * obj_mask) / B

        total_loss = loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_cls
        return total_loss

if __name__ == "__main__":
    from config import config
    cfg = config.Config()
    criterion = YOLOv2Loss(cfg)

    pred = torch.rand(2,13,13,5,9)
    target = torch.rand(2,13,13,5,9)
    target[0, 6, 1, 0] = torch.tensor([0.5, 0.5, 0.2, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0])  # 模拟一个物体

    loss = criterion(pred, target)
    print("总 Loss:", loss.item())