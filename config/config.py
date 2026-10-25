class Config:
    train_images_dir = "./HelmetDataset-YOLO-Train/images"
    train_labels_dir = "./HelmetDataset-YOLO-Train/labels"
    val_images_dir = "./HelmetDataset-YOLO-Val/images"
    val_labels_dir = "./HelmetDataset-YOLO-Val/labels"
    ckpt_dir = "./checkpoints" # 存模型权重
    output_dir = "./output" # 存检测结果图

    img_size = 416 # v2 标准输入尺寸
    num_classes = 4
    class_names = ['motor', 'no helmet', 'number', 'with helmet']
    num_workers = 4

    # anchor（用 kmeans_anchors.py 聚类出来的结果，网格单位）
    # anchor 表 (B, 2)，每行是 (pw, ph)（‌网格单位‌）。
    anchors = [
        [1.26, 1.14],   # 最小的 Anchor
        [3.49, 0.70],   # 扁长形 Anchor
        [2.66, 1.99],   # 偏方形小 Anchor
        [5.19, 6.89],   # 中等偏大 Anchor
        [10.53, 8.55]   # 最大的 Anchor
    ]

    # 网格
    stride = 32
    grid_size = img_size // stride  # 13
    num_anchors = len(anchors)

    # 训练
    batch_size = 10
    epochs = 100
    lr = 1e-4
    weight_decay = 1e-4 # 权重衰减
    lr_scheduler = 'cosine'
    lambda_coord = 0.5 # 坐标损失权重
    lambda_obj = 5.0
    lambda_noobj = 0.01 # 无目标网格置信度损失权重
    obj_conf_threshold = 0.1 # 正样本置信度目标 ; 阈值（Thresholding）
    noobj_conf_thresh = 0.0
    ignore_thresh = 0.5 # IoU 大于此但不是最佳 anchor，忽略
    nms_iou_thresh = 0.5
    conf_thresh = 0.12  # 推理置信度阈值
    early_stop_patience = 20