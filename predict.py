import torch
import torchvision
from torchvision import transforms
from PIL import Image

from config.config import Config
from data.dataset import DatasetV2
from models.YoloV2 import YOLOv2

cfg = Config()
dataset = DatasetV2(images_dir=cfg.val_images_dir,labels_dir=cfg.val_labels_dir,cfg=cfg)
to_tensor = transforms.ToTensor()

image_path = r".\HelmetDataset-YOLO-Val\images\new89.png"
label_path = r".\HelmetDataset-YOLO-Val\labels\new89.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLOv2(num_classes=cfg.num_classes,num_anchors=cfg.num_anchors)
model.to(device)
model.load_state_dict(torch.load("checkpoints/yolov2_best_model.pth"))
model.eval()

image = Image.open(image_path).convert('RGB')
with open(label_path, 'r') as f:
    lines = f.readlines()
    boxes = [line.strip().split(" ") for line in lines]
    boxes = [list(map(float, i)) for i in boxes]
    cls_ids = [int(i.pop(0)) for i in boxes]
    boxes = torch.tensor(boxes)
image, target,scale,pad_w,pad_h = dataset.letterbox(image,boxes,return_meta=True)

image_to_tensor = to_tensor(image) # torch.Size([3, 416, 416])
# print(image) # <PIL.Image.Image image mode=RGB size=416x416 at 0x1926F2DDE00>
# print(target) # -> (3,4)
image_to_tensor = image_to_tensor.unsqueeze(0) # torch.Size([1, 3, 416, 416])
image_to_tensor = image_to_tensor.to(device)

pred = model(image_to_tensor)
print(pred.shape)

pred_xy = torch.sigmoid(pred[..., 0:2])
print(pred_xy.shape)
pred_wh = pred[...,2:4]
pred_conf = torch.sigmoid(pred[..., 4])
print("模型预测的最大置信度:", pred_conf.max().item())
pred_cls = torch.softmax(pred[...,5:],dim=-1)

# 还原到 416x416 特征图坐标
stride = cfg.stride

# 网格坐标
grid_size = cfg.grid_size
grid_x = torch.arange(grid_size).repeat(grid_size,1).view(1,13,13,1,1).float().to(device)
grid_y = torch.arange(grid_size).t().repeat(grid_size,1).view(1,13,13,1,1).float().to(device)

# 真实中心坐标
# 使用 [0:1] 而不是 0 是为了保持维度
print(pred_xy[...,0:1].shape)
print(grid_x.shape)
cx = (pred_xy[...,0:1] + grid_x) * stride
cy = (pred_xy[...,1:2] + grid_y) * stride

# 真实宽高 需要乘以 Anchor
"""
预测值是锚框宽度的倍数，本质上是‌将绝对尺寸的回归问题转化为相对比例的回归问题‌。
公式：b_w = p_w * exp(t_w)
含义：真实宽度 = 锚框宽度 × 缩放因子。
注意：
[:,0] != [...,0]
>>> x1 = torch.rand(2,3,4)
>>> x1
tensor([[[0.0880, 0.2047, 0.0087, 0.7046],
         [0.5605, 0.5597, 0.5395, 0.5817],
         [0.8128, 0.5012, 0.0388, 0.2243]],

        [[0.5566, 0.1262, 0.4483, 0.0305],
         [0.9300, 0.1640, 0.7667, 0.9255],
         [0.0565, 0.5893, 0.2174, 0.1171]]])
>>> x1[:,0] # shape: torch.Size([2, 4])
tensor([[0.0880, 0.2047, 0.0087, 0.7046],
        [0.5566, 0.1262, 0.4483, 0.0305]])
>>> x1[...,0] # shape: torch.Size([2, 3])
tensor([[0.0880, 0.5605, 0.8128],
        [0.5566, 0.9300, 0.0565]])
"""
anchors = torch.tensor(cfg.anchors).to(device) # (5, 2)
w = torch.exp(pred_wh[...,0:1]) * anchors[:,0].view(1,1,1,cfg.num_anchors,1) # num_anchors = len(anchors)
h = torch.exp(pred_wh[...,1:2]) * anchors[:,1].view(1,1,1,cfg.num_anchors,1)

x1 = cx - w / 2
y1 = cy - h / 2
x2 = cx + w / 2
y2 = cy + h / 2

# 过滤与 NMS（非极大值抑制）
conf_mask = pred_conf > cfg.conf_thresh
# 把 x1, y1, x2, y2 展平，并且只保留高置信度的框
boxes = torch.cat([x1,y1,x2,y2],dim=-1)[conf_mask]
scores = pred_conf[conf_mask]
classes = torch.argmax(pred_cls,dim=-1)[conf_mask]
# 2. NMS去重 (一个目标可能被多个网格预测)
keep_indices = torchvision.ops.nms(boxes,scores,iou_threshold=cfg.nms_iou_thresh)
boxes = boxes[keep_indices]
scores = scores[keep_indices]
classes = classes[keep_indices]
print(f"最终保留了 {len(boxes)} 个框")


# AI
from PIL import ImageDraw, ImageFont
import os

# 1. 重新用 PIL 读取原图 (因为之前你读的 image 已经被 letterbox 变成 416x416 了)
# 注意：你的原始 image_path 是 png
img_pil = Image.open(image_path).convert('RGB')
draw = ImageDraw.Draw(img_pil)

# 2. 尝试加载字体，让文字大一点
try:
    # Windows 系统通常有 arial.ttf，Mac 可以用 Arial.ttf 或其他
    # 如果报错找不到字体，可以随便换一个你系统里有的 ttf 文件路径
    font = ImageFont.truetype("arial.ttf", size=15)
except IOError:
    print("⚠️ 找不到 arial.ttf，使用默认字体（可能会比较小）")
    font = ImageFont.load_default()

# 3. 遍历框并画图
for box, score, cls_id in zip(boxes, scores, classes):
    # box 是 416 尺度下的坐标 [x1, y1, x2, y2]
    box = box.cpu().detach().numpy()

    # 减去 padding 且 除以 scale，还原到原图坐标
    x1 = int((box[0] - pad_w) / scale)
    y1 = int((box[1] - pad_h) / scale)
    x2 = int((box[2] - pad_w) / scale)
    y2 = int((box[3] - pad_h) / scale)

    # 防止坐标越界（PIL 如果画出界不会报错，但画不全）
    img_w, img_h = img_pil.size
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    # 画矩形框 (RGB 格式：绿色)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

    # 画文字背景底色（可选，为了让文字更清晰）
    label = f"{cfg.class_names[cls_id]} {score:.2f}"
    text_bbox = draw.textbbox((x1, y1 - 15), label, font=font)
    draw.rectangle(text_bbox, fill=(0, 255, 0)) # 绿色底色

    # 画黑色文字
    draw.text((x1, y1 - 15), label, fill=(0, 0, 0), font=font)

# 4. 保存或显示图片
save_path = "outputs/result_pil.jpg"
img_pil.save(save_path)
print(f"✅ 检测结果已保存到 {save_path}")

# 如果你在本地运行，可以直接弹窗查看
img_pil.show()