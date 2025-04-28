import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set seed for reproducibility
np.random.seed(42)

# Parameters
n_points   = 100
covariance = 0.075 * np.eye(2)
mean1      = np.array([-0.35, 0.0])
mean2      = np.array([ 0.35, 0.0])

# Generate two overlapping clusters
c1 = np.random.multivariate_normal(mean1, covariance, n_points)
c2 = np.random.multivariate_normal(mean2, covariance, n_points)

# Stack and run KMeans
data  = np.vstack((c1, c2))
kmeans = KMeans(n_clusters=2, random_state=42).fit(data)
centers = kmeans.cluster_centers_
labels  = kmeans.labels_

# For each cluster, find the 15 nearest points to its center
k = 15
nearest_points = []
for i in range(2):
    pts = data[labels == i]
    d   = np.linalg.norm(pts - centers[i], axis=1)
    nearest_points.append(pts[np.argsort(d)[:k]])

# Plot
plt.figure(figsize=(8,6))
plt.scatter(data[:,0], data[:,1], color='grey', alpha=0.6)

colors = ['blue','red']
for i in range(2):
    np_pts = nearest_points[i]
    plt.scatter(np_pts[:,0], np_pts[:,1],
                color=colors[i], alpha=0.6,
                label=f'{k} nearest to center {i}')
    plt.scatter( centers[i,0], centers[i,1],
                 marker='x', color=colors[i],
                 s=100, label=f'Center {i}' )

plt.title("k‑Means Anchors as Linear Combinations of the n Closest Points")
plt.xlabel("X"); plt.ylabel("Y")
plt.gca().set_aspect('equal','box')
plt.grid(True)
# plt.legend()
plt.show()
