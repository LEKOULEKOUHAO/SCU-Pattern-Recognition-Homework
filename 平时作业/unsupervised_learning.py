import random
import math
import matplotlib.pyplot as plt
import numpy as np

def generate_data(num_samples, means, stds):
    """生成高斯分布数据"""
    data = []
    for i in range(num_samples):
        # 随机选择一个高斯分布
        distribution = random.choice(range(len(means)))
        # 生成样本点
        x = random.gauss(means[distribution][0], stds[distribution][0])
        y = random.gauss(means[distribution][1], stds[distribution][1])
        # 将数据限制在0-10之间
        x = max(0, min(x, 10))
        y = max(0, min(y, 10))
        data.append([x, y])
    return data

# 设置两个高斯分布的参数
means = [[2, 2], [8, 8]]
stds = [[1, 1], [1, 1]]
data = generate_data(100, means, stds)

# 将数据转换为numpy数组
data = np.array(data)

# K-Means 实现
def kmeans(data, k, max_iterations=100):
    # 随机选择k个中心点
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        # 计算每个点到中心点的距离，并将其分配到最近的簇
        distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        # 计算新的中心点
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        # 如果中心点不再变化，则停止迭代
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

labels_kmeans, centroids_kmeans = kmeans(data, 2)

# DBSCAN 实现
def dbscan(data, eps, min_samples):
    labels = np.full(len(data), -1)  # 初始化所有点为噪声
    cluster_id = 0
    for i in range(len(data)):
        if labels[i] != -1:
            continue
        neighbors = []
        for j in range(len(data)):
            distance = math.dist(data[i], data[j]) # 计算欧几里得距离
            if distance <= eps:
                neighbors.append(j)
        if len(neighbors) < min_samples:
            labels[i] = -1 #噪声点
        else:
            cluster_id += 1
            expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples)
    return labels

def expand_cluster(data, labels, i, neighbors, cluster_id, eps, min_samples):
    labels[i] = cluster_id
    seeds = neighbors.copy()
    while seeds:
        j = seeds.pop(0)
        if labels[j] == -1:
            labels[j] = cluster_id
            new_neighbors = []
            for k in range(len(data)):
                distance = math.dist(data[j], data[k])
                if distance <= eps:
                    new_neighbors.append(k)
            if len(new_neighbors) >= min_samples:
                seeds.extend(new_neighbors)

labels_dbscan = dbscan(data, eps=1.5, min_samples=5) # 需要根据数据调整eps和min_samples

# 结果绘制
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=labels_kmeans)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], c=labels_dbscan)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()