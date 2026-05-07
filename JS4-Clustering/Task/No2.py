import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

# Membuat data contoh
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60,
                  random_state=0)

# Menggunakan DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=10)
labels = dbscan.fit_predict(X)

# Visualisasi hasil clustering
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title("Hasil DBSCAN Clustering")
plt.show()