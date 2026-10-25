import glob
import numpy as np
from config import config

config = config.Config()

class KMeansAnchors(object):
    def __init__(self,cfg):
        self.grid_size = cfg.grid_size
        self.train_labels_dir = cfg.train_labels_dir

    def load_dataset(self, labels_dir_name,grid_size):
        labels_name = glob.glob("." + labels_dir_name + "/*.txt")
        w_h_data = []
        for label in labels_name:
            with open(label, "r") as f:
                lines = f.readlines()
                for line in lines:
                    w,h = list(map(float,line.strip().split(" ")[-2:]))
                    w_h_data.append([w * grid_size,h * grid_size])
        return np.array(w_h_data)

    def iou(self,box,clusters):
        """
        计算一个框 box 和所有簇中心 clusters 的 IoU
        box: (w, h)
        clusters: (k, 2)
        """
        inter_w = np.minimum(clusters[0],box[:,0])
        inter_h = np.minimum(clusters[1],box[:,1])
        inter_area = inter_w * inter_h
        box_area = box[:, 0] * box[:, 1]
        cluster_areas = clusters[0] * clusters[1]
        union_area = box_area + cluster_areas - inter_area
        # 加上 1e-6 防止除以 0
        return inter_area / (union_area + 1e-6)

    def kmeans_anchors(self,boxes,k,max_iter=100):
        n = boxes.shape[0] # 几个框
        centroids = boxes[np.random.choice(n,k,replace=False)] # 随机挑选 k 个框作为初始 centroids
        old_centroids = np.zeros(centroids.shape)

        for iter_ in range(max_iter):
            distance = np.zeros((n,k))
            for i in range(k):
                distance[:,i] = 1.0 - self.iou(boxes, centroids[i])
            # 找到每个 box 距离最近的 centroid 索引
            cluster_index = np.argmin(distance,axis=1)
            # 深拷贝一下，防止改乱了内存
            # old_centroids = centroids.copy()

            for i in range(k):
                # 找出所有被分配到第 i 个簇的框
                cluster_boxes = boxes[cluster_index == i]

                # 如果这个簇里有框，就求均值；如果空的就保留原样（防止除以 0）
                if len(cluster_boxes) > 0:
                    centroids[i] = np.mean(cluster_boxes, axis=0)
            # np.allclose 是比较两个数组是否非常接近（允许浮点数微小误差）
            if np.allclose(old_centroids, centroids):
                print(f"迭代 {iter_} 次后收敛！")
                break
        # 3. 最后，把 5 个 Anchor 按面积从小到大排序（YOLO惯例）
        # 面积 = 宽 * 高
        areas = centroids[:, 0] * centroids[:, 1]
        # argsort 返回从小到大排序的索引
        sort_indices = np.argsort(areas)
        centroids = centroids[sort_indices]
        return centroids

if __name__ == "__main__":
    kma = KMeansAnchors(config)
    data = kma.load_dataset(config.train_labels_dir,config.grid_size)
    centroid = kma.kmeans_anchors(data, 5, max_iter=100)
    print(centroid)