import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import matplotlib.pyplot as plt
import numpy as np
import imageio
from tqdm import tqdm

# Generate 100 black points from N(0,1)
np.random.seed(0)
black_points = np.random.randn(100, 2)
red_points = np.array([[ 1.5,  1], [0.1, 0.15]])
n_frames      = 200            # number of frames in the animation
output_gif    = 'visualization_for_presentaion/cosine_sim_rotation_scale.gif'
fps           = 20
fig, ax = plt.subplots(figsize=(6, 6))

# 1) plot the black points
ax.scatter(black_points[:, 0], black_points[:, 1],
           s=20, label='Black points', c="C0")
ax.scatter(red_points[:, 0], red_points[:, 1],
           s=20, label='Black points', c="C0")

ax.axhline(0, color='grey', linewidth=1)
ax.axvline(0, color='grey', linewidth=1)
ax.set_aspect('equal', 'box')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xticks(np.arange(-2, 3))
ax.set_yticks(np.arange(-2, 3))
plt.savefig("visualization_for_presentaion/cosine_1.png")



fig, ax = plt.subplots(figsize=(6, 6))

# 1) plot the black points
ax.scatter(black_points[:, 0], black_points[:, 1],
           s=20, label='Black points', c="C0")
ax.scatter(red_points[:, 0], red_points[:, 1],
           s=40, label='Black points', c="red")

ax.axhline(0, color='grey', linewidth=1)
ax.axvline(0, color='grey', linewidth=1)
ax.set_aspect('equal', 'box')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xticks(np.arange(-2, 3))
ax.set_yticks(np.arange(-2, 3))
plt.savefig("visualization_for_presentaion/cosine_2.png")

# 1) plot the black points
ax.scatter(black_points[:, 0], black_points[:, 1],
           s=20, c="C0")
ax.scatter(red_points[:, 0], red_points[:, 1],
           s=40, label='Black points', c="red")
for x, y in red_points:
    ax.arrow(0, 0, x, y,
             head_width=0.05, head_length=0.1,
             length_includes_head=True,
             color='red')
ax.axhline(0, color='grey', linewidth=1)
ax.axvline(0, color='grey', linewidth=1)
cosine = np.dot(red_points[0], red_points[1])/(np.linalg.norm(red_points[0]) * np.linalg.norm(red_points[1]))
ax.set_title(f"Cosine similarity: {cosine:.3f}")
ax.set_aspect('equal', 'box')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xticks(np.arange(-2, 3))
ax.set_yticks(np.arange(-2, 3))
plt.savefig("visualization_for_presentaion/cosine_3.png")