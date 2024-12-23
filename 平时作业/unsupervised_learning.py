import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成高斯分布数据
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=42)
# K-Means 实现
class KMeans:
    def __init__(self, n_clusters=2, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
    def fit(self, X):
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        for _ in range(self.max_iters):
            distances = self._compute_distances(X)
            labels = np.argmin(distances, axis=1)
            self.centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return labels
    def _compute_distances(self, X):
        return np.array([[np.linalg.norm(x - centroid) for centroid in self.centroids] for x in X])
# 运行 K-Means
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit(X)
# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering")
plt.show()

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        labels = -np.ones(X.shape[0])
        cluster_id = 0
        for i in range(X.shape[0]):
            if labels[i] != -1:
                continue
            neighbors = self._region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise
            else:
                self._expand_cluster(X, labels, i, neighbors, cluster_id)
                cluster_id += 1
        return labels

    def _region_query(self, X, point_idx):
        neighbors = []
        for i in range(X.shape[0]):
            if np.linalg.norm(X[point_idx] - X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def _expand_cluster(self, X, labels, point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            i += 1

# 运行DBSCAN
dbscan_custom = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan_custom = dbscan_custom.fit_predict(X)
# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan_custom, s=30, cmap='viridis')
plt.title("Custom DBSCAN Clustering")
plt.show()