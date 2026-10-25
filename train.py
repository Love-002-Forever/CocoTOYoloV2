import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import config
from data.dataset import DatasetV2
from models.YoloV2 import YOLOv2
from loss.loss import YOLOv2Loss

def train():
    cfg = config.Config()
    summary = SummaryWriter("./logs")
    train_loss_history = []
    val_loss_history = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 加载数据
    train_dataset = DatasetV2(cfg.train_images_dir, cfg.train_labels_dir, cfg)
    val_dataset = DatasetV2(cfg.val_images_dir, cfg.val_labels_dir, cfg)
    train_data_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = YOLOv2(num_classes=cfg.num_classes,num_anchors=cfg.num_anchors)
    model = model.to(device)
    criterion = YOLOv2Loss(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr) # 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch_model = 0
    for epoch in range(cfg.epochs):
        print('Epoch {}/{}'.format(epoch, cfg.epochs))
        print('-' * 10)

        epoch_loss = 0
        for images, labels in train_data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()
            # 向前传播
            pred = model(images)
            # 计算损失
            loss = criterion(pred, labels)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()

            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_data_loader)
        summary.add_scalar("train_loss", avg_train_loss, epoch)
        train_loss_history.append(avg_train_loss) # 将此次 loss 存入

        model.eval() # 告诉网络：关掉 Dropout 和 BatchNorm 的训练模式
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_data_loader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images)
                loss = criterion(pred, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_data_loader)
        summary.add_scalar("val_loss", avg_val_loss, epoch)
        val_loss_history.append(avg_val_loss)

        # 更新学习率lr
        scheduler.step()

        print(f"Epoch {epoch + 1} 结束\n平均 Train_Loss: {avg_train_loss:.4f}\n平均 Val_Loss: {val_loss:.4f}")

        # 早停，保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_epoch_model = epoch
            torch.save(model.state_dict(),f"./checkpoints/yolov2_best_model.pth")
            print(f"发现最佳模型在 {best_epoch_model} Epoch，已保存！")
        else:
            patience_counter += 1
            if patience_counter > cfg.early_stop_patience:
                print(f"⚠️ 连续多个 Epoch Val Loss ({cfg.early_stop_patience}次)未下降，触发早停！\n最好模型在第 {best_epoch_model} Epoch")
                break


if __name__ == '__main__':
    train()