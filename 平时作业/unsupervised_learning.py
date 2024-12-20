import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
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

from sklearn.cluster import DBSCAN
# 运行 DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X)
# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels_dbscan, s=30, cmap='viridis')
plt.title("DBSCAN Clustering")
plt.show()