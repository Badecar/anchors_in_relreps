import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import numpy as np
import matplotlib.pyplot as plt
import imageio
from anchors import get_optimized_anchors
import random
from scipy.spatial.distance import pdist
from tqdm import tqdm
import torch

# === Parameters ===
def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_random_seeds(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

num_anchors = 2
reps = 8
black_points = np.random.randn(1000, 2).astype(np.float32)
black_points = black_points - np.mean(black_points, axis=0)
red_points    = black_points[np.random.choice(np.arange(len(black_points)), 10)]

n_frames      = 100            # number of frames in the animation
output_gif    = 'visualization_for_presentaion/animation_P.gif'
fps           = 20

# Create a directory for frame images
frames_dir = 'visualization_for_presentaion/gif_frames'
os.makedirs(frames_dir, exist_ok=True)
frame_paths = []

# === Animation: Scaling the red points from 0.2× to 2.0× their original distance from the origin ===

black_points  # (n, d)
indices = np.arange(len(black_points))
# Normalize embeddings once for cosine computations.
norms = np.linalg.norm(black_points, axis=1, keepdims=True)
normalized_embeddings = black_points
anchors_list = []

for i in tqdm(range(100), desc="getting anchors"):
    set_random_seeds(0)
    _, P_anchors_list = get_optimized_anchors(
        emb = [black_points],
        anchor_num=2,
        epochs=i*2,
        lr=1e-2,
        coverage_weight=0.95,
        diversity_weight=0.05,
        exponent=1,
        dist_measure="euclidean", ## "euclidean", "mahalanobis", "cosine"
        verbose=False,
        device=device,
    )
    anchors_list.append(P_anchors_list[0])

for i, s in tqdm(enumerate(anchors_list), desc="generating frames"):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=40, c="black")
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)

    ax.scatter(s[0][0],   s[0][1],   c='red', s=80)
    ax.scatter(s[1][0],   s[1][1],   c='red', s=80)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    
    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)

# === Combine frames into a GIF ===
frames = [imageio.imread(fp) for fp in frame_paths]
imageio.mimsave(output_gif, frames, fps=fps, loop=0)

# === Cleanup frames directory (optional) ===
for fp in frame_paths:
    os.remove(fp)

print(f"GIF written to {output_gif}")