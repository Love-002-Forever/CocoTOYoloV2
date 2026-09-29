import os
from pycocotools.coco import COCO

class CocoToYolo(object):
    def __init__(self):
        self.annotations_path = r".\annotations_trainval2017\annotations"
        self.instances_train_path = os.path.join(self.annotations_path, "instances_train2017.json")
        self.instances_val_path = os.path.join(self.annotations_path, "instances_val2017.json")
        self.train_images_path = r".\train2017\train2017" # 此文件夹内全为图片
        self.train_labels_path = r".\train2017\labels"
        self.classes_path = os.path.join(r".\train2017", "classes.txt")
        self.coco_train = COCO(self.instances_train_path)

    def load(self):
        cats_ids = self.coco_train.getCatIds()
        self.cat_id_to_index = {cid:i for i,cid in enumerate(cats_ids)} # 为 id 做映射

        classes = self.coco_train.loadCats(cats_ids)  # [{'supercategory': 'xxxxx', 'id': 1, 'name': 'xxx'},...]
        classes_list = self.create_classes_txt(self.classes_path,classes)

        train_ids = self.coco_train.getImgIds()
        self.create_labels_txt(train_ids)


    def create_classes_txt(self,classes_path,classes):
        # 创建classes.txt文件并创建 classes 列表
        classes_list = []
        if not os.path.isfile(classes_path):
            print("已创建 classes.txt \npath:{}".format(classes_path))
            with open(classes_path, "w+") as f:
                for cat in classes:
                    f.write(cat["name"] + "\n")
                    classes_list.append(cat["name"])
        else:
            print("classes.txt 文件存在无更新")
            with open(classes_path, "r") as f:
                classes_list = f.read().splitlines()
        return classes_list


    def create_labels_txt(self,ids):
        if not os.path.isdir(self.train_labels_path):
            os.makedirs(self.train_labels_path) # 检查是否存在labels文件夹
            print("labels文件夹已创建")
        len_ids = len(ids)
        for i,idx in enumerate(ids):
            print("{:-^20}".format(str(i/len_ids*100) + "%"))
            img = self.coco_train.loadImgs(idx)
            image_id = img[0]["id"]
            label_name = "{:0>12}.txt".format(image_id)
            label_path = os.path.join(self.train_labels_path, label_name)  # label文件夹 路径
            # print(img)
            total_width, total_height = img[0]["width"], img[0]["height"]
            ann_id = self.coco_train.getAnnIds(imgIds=idx) # 参数 iscrowd 能清晰画出轮廓的标 0，画不出来的一堆东西标 1 并压缩成 RLE，
            ann = self.coco_train.loadAnns(ann_id)
            lines = []
            for obj in ann:
                if obj.get("iscrowd", 0) == 1 :continue # 过滤密集标签
                classes_id = self.cat_id_to_index[obj["category_id"]]
                bbox = obj["bbox"] #coco x y width height
                x_center,y_center,width,height = self.coco_coord_to_yolo_coord(bbox[0],bbox[1],bbox[2],bbox[3],total_width,total_height)
                lines.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(classes_id,x_center,y_center,width,height))
                # # 判断 label文件是否存在，判断用 w+ 还是 a+ (废弃代码，效率低速率慢)
                # if not os.path.isfile(label_path):
                #     with open(label_path, "w+") as f:
                #         f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(classes_id,x_center,y_center,width,height)) # yolo 类别id x_center y_center width height
                # else:
                #     with open(label_path, "a+") as f:
                #         f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(classes_id,x_center,y_center,width,height))

            # 【关键修复】无论有没有标注，都写入文件
            # 如果没有标注，lines 为空，写入空文件（YOLO 视为背景图/负样本）
            with open(label_path, "w") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")


    def coco_coord_to_yolo_coord(self,x,y,w,h,tw,th): # tw,th表示图像总宽，高
        x_center_px = (x + w / 2.0)
        y_center_px = (y + h / 2.0)
        x_center_norm = x_center_px / tw
        y_center_norm = y_center_px / th
        w_norm = w / tw
        h_norm = h / th

        return x_center_norm,y_center_norm,w_norm,h_norm

if __name__ == "__main__":
    load = CocoToYolo()
    load.load()