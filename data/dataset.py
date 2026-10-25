import os
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms
from PIL import Image
from config import config

# cfg = config.Config()

class DatasetV2(Dataset):
    def __init__(self,images_dir,labels_dir,cfg,augment=False):
        super(DatasetV2,self).__init__()
        # 废弃 self.images_path = list(os.path.join(images_dir,os.path.splitext(i)[0] + ".jpg") for i in sorted(os.listdir(labels_dir)))
        self.images_path = []
        self.labels_dir = labels_dir
        self.cfg = cfg
        self.augment = augment
        anchors = torch.tensor(cfg.anchors)
        self.anchors = anchors.detach().clone()
        self.to_tensor = transforms.ToTensor()

        # 从 .txt 文件找对应图片； 不对找不到的图片做处理
        label_files = sorted(os.listdir(labels_dir))
        for label_file in label_files:
            if not label_file.endswith('.txt'):
                continue

            base_name = os.path.splitext(label_file)[0]
            for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]:
                image_path = os.path.join(images_dir,base_name+ext)
                if os.path.exists(image_path):
                    self.images_path.append(image_path)
        print(f"✅ 成功加载 {len(self.images_path)} 个样本。")


    def __len__(self):
        return len(self.images_path)

    def letterbox(self,image,boxes,return_meta=False):
        # 判断是图片地址还是图片tensor
        if isinstance(image, str):
            # 等比例缩放 + 灰边填充，返回新图和变换后的框; 返回的框是"在 416×416 新图上的像素坐标"
            image = Image.open(image).convert('RGB')

        image_w = image.size[0]
        image_h = image.size[1]

        # 缩放比：目标 / 原图，取 min 保证不超出
        scale = min(self.cfg.img_size/image_w,self.cfg.img_size/image_h)
        new_w = round(image_w * scale)
        new_h = round(image_h * scale)

        pad_w = (self.cfg.img_size - new_w) / 2 # 均分到左右(填多少灰边)
        pad_h = (self.cfg.img_size - new_h) / 2 # 上下

        image = image.resize((new_w,new_h))
        # 新建一个灰画布; canvas: 画布
        canvas = Image.new('RGB',(self.cfg.img_size,self.cfg.img_size),color=(114,114,114)) # RGB 值 114,114,114 是 YOLO 官方用的灰
        canvas.paste(image,(int(pad_w),int(pad_h))) # 把resize的图居中贴放在画布上

        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        x1 = x1 * scale + pad_w
        y1 = y1 * scale + pad_h
        x2 = x2 * scale + pad_w
        y2 = y2 * scale + pad_h

        if return_meta:
            return canvas,torch.stack([x1,y1,x2,y2],dim=-1),scale,pad_w,pad_h
        return canvas,torch.stack([x1,y1,x2,y2],dim=-1) # 返回新的画布及缩放后的像素框

    def encode_boxes(self,boxes,cls_ids):
        target = torch.zeros(self.cfg.grid_size, self.cfg.grid_size, self.cfg.num_anchors , 5 + self.cfg.num_classes)

        for box,cls_id in zip(boxes,cls_ids):
            x1, y1, x2, y2 = box

            # 转换成 YOLO 坐标
            cx_px = (x1 + x2) / 2
            cy_px = (y1 + y2) / 2
            w_px = x2 - x1
            h_px = y2 - y1

            # 转网格(将中心点坐标转换成网格坐标)
            cx = cx_px / self.cfg.stride
            cy = cy_px / self.cfg.stride
            w = w_px / self.cfg.stride
            h = h_px / self.cfg.stride

            # 中心点坐标在哪个格子上
            grid_cx = int(cx) # floor，列号（x）→ 范围 0~12
            grid_cy = int(cy) # floor，行号（y）→ 范围 0~12
            # ‌必须防越界‌：如果物体中心正好贴右边，cx 可能 = 13，int(13) = 13 直接越界崩。
            grid_cx = min(grid_cx,self.cfg.grid_size - 1)
            grid_cy = min(grid_cy,self.cfg.grid_size - 1)

            IoU = []
            pw_all,ph_all = self.anchors[...,0],self.anchors[...,1]
            for i in range(self.cfg.num_anchors):
                intersection = torch.min(w, pw_all[i]) * torch.min(h, ph_all[i]) # 交集
                union = w * h + pw_all[i] * ph_all[i] - intersection # 并集
                IoU.append(intersection / union)
            b = torch.stack(IoU).argmax().item()
            print(IoU)
            print("b", b)
            # 取出最大 IoU 的 anchor 的宽高
            pw = pw_all[b]
            ph = ph_all[b]

            # 算 4 个目标
            tx = cx - grid_cx # 格内横向偏移，天然 0~1
            ty = cy - grid_cy # 格内纵向偏移
            w = torch.clamp(w,1e-6)
            h = torch.clamp(h,1e-6)
            # 宽度缩放比取对数
            tw = torch.log(w/pw)
            th = torch.log(h/ph)

            # 【关键修复】所有索引变量显式转为long类型，符合PyTorch索引要求
            grid_cx = torch.tensor(grid_cx, dtype=torch.long)
            grid_cy = torch.tensor(grid_cy, dtype=torch.long)
            b = torch.tensor(b, dtype=torch.long)
            cls_id = torch.tensor(cls_id, dtype=torch.long)

            target[grid_cx,grid_cy,b,0] = tx
            target[grid_cx,grid_cy,b,1] = ty
            target[grid_cx,grid_cy,b,2] = tw
            target[grid_cx,grid_cy,b,3] = th
            target[grid_cx,grid_cy, b, 4] = 1  # objectness
            target[grid_cx,grid_cy, b, 5 + cls_id] = 1  # one-hot 类别

        return target

    def coord_trans(self,coord:torch.Tensor,img_size):
        cx,cy,w,h = coord[...,0],coord[...,1],coord[...,2],coord[...,3]
        img_w,img_h = img_size

        cx_px = cx * img_w
        cy_px = cy * img_h
        w_px = w * img_w
        h_px = h * img_h

        x1 = cx_px - w_px / 2
        y1 = cy_px - h_px / 2
        x2 = cx_px + w_px / 2
        y2 = cy_px + h_px / 2

        stack_coord = torch.stack([x1,y1,x2,y2],dim=-1)

        return stack_coord

    def __getitem__(self, index):
        img_path = self.images_path[index]
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert('RGB')
        target_path = os.path.join(self.labels_dir,name + ".txt")
        with open(target_path,'r') as f:
            lines = f.readlines()
            boxes = [line.strip().split(" ") for line in lines]
            boxes = [list(map(float,i)) for i in boxes]
            cls_ids = [int(i.pop(0)) for i in boxes]
            boxes = torch.tensor(boxes)
            # print(boxes,cls_ids)
        trans_xywh_to_xyxy = self.coord_trans(boxes,img.size)
        canvas,boxes_pixel_new = self.letterbox(img,trans_xywh_to_xyxy)
        target = self.encode_boxes(boxes_pixel_new,cls_ids)
        return self.to_tensor(canvas),target #,name + '.jpg'


if __name__=="__main__":
    import os
    file_path:str = os.path.abspath(__file__) # 1. 获取当前文件的绝对路径
    dir_path = os.path.dirname(file_path) # 2. 获取当前文件所在目录
    parent_dir = os.path.dirname(dir_path) # 3. 获取上一级目录 (当前目录的父目录)
    cfg = config.Config()
    path_images = os.path.join(parent_dir,"HelmetDataset-YOLO-Train/images")
    path_labels = os.path.join(parent_dir,"HelmetDataset-YOLO-Train/labels")
    img = os.path.join(path_images,"new1.jpg")
    print(img)
    x_center,y_center,w,h = 0.560071,0.087838,0.30742,0.141892
    img2 = Image.open(img).convert('RGB')
    img_w,img_h = img2.size

    cx_px = x_center * img_w
    cy_px = y_center * img_h
    w_px = w * img_w
    h_px = h * img_h

    x1 = cx_px - w_px / 2
    y1 = cy_px - h_px / 2
    x2 = cx_px + w_px / 2
    y2 = cy_px + h_px / 2
    print("="*20)
    print(x1,y1,x2,y2)
    print("="*20)

    V2 = DatasetV2(path_images,path_labels,cfg=cfg,augment=True)
    print(V2.letterbox(img,torch.tensor([[x1,y1,x2,y2]])))

    print("=" * 20)
    ds = DatasetV2(path_images,path_labels,cfg=cfg,augment=True)
    img, target = ds[0]
    print(img.shape)
    print(target.shape)
    # 打印所有非零位置
    idx = (target[..., 4] == 1).nonzero()   # 找出正样本位置
    print(idx)                              # 应该是 (i, j, b) 表示第 i 列、第 j 行、第 b 个Anchor(匹配到了正样本)
    print(target[idx[0][0], idx[0][1], idx[0][2]])  # 打印那 9 个数