import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import numpy as np
import matplotlib.pyplot as plt
import imageio
from anchors import greedy_one_at_a_time_single_euclidean
import random
from scipy.spatial.distance import pdist
from tqdm import tqdm

# === Parameters ===
np.random.seed(0)
random.seed(0)
num_anchors = 2
reps = 8
black_points = np.random.randn(1000, 2)
black_points = black_points - np.mean(black_points, axis=0)
red_points    = np.array([[ 1,  1],
                          [-1, -1]])
n_frames      = 15            # number of frames in the animation
output_gif    = 'visualization_for_presentaion/animation_greedy_search.gif'
fps           = 1

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
anchors_list_for_plot = []


# Function to compute coverage using pdist on the anchor set.
def compute_coverage(anchor_array):
    # anchor_array should have shape (m, d) where m is small.
    # pdist returns pairwise distances; raise each to the given exponent and sum.
    if len(anchor_array) == 2:
        return np.sum(pdist(anchor_array, metric="euclidean"))
    print("test")
    quit()
# Greedily add anchors.
for _ in range(num_anchors*reps - 1):
    best_score = np.inf
    previos_best = np.inf
    best_index = None
    deleted = None
    best_new_min_dists = None

    # Evaluate each candidate index that is not already selected.
    for candidate in indices:
        if any(np.all(pt == black_points[candidate]) for pt in anchors_list):
            continue

        # Get candidate's normalized vector.
        # Compute cosine distances from all embeddings to this candidate in a vectorized way.
        candidate_dists_list = np.linalg.norm(black_points - black_points[candidate], axis=1)
        # New minimum distances if candidate were added.
        if len(anchors_list) > 0:
            new_min_dists = np.minimum(min_dists, candidate_dists_list)
            # Compute coverage for anchors + candidate.
            current_anchor_vectors = np.array(anchors_list)  # shape (m, d)
            candidate_anchor_array = np.vstack([current_anchor_vectors, black_points[candidate]])
            diversity_val = np.mean(new_min_dists, axis=0)  # diversity is the average min distance.
            coverage_val = compute_coverage(candidate_anchor_array)
            # Overall objective: maximize diversity while penalizing coverage.
            current_score = 0.95 * diversity_val - 0.05 * coverage_val

            if current_score < best_score:
                best_score         = current_score
                best_index         = candidate
                best_new_min_dists = new_min_dists

        else:
            new_min_dists = np.abs(candidate_dists_list)
            candidate_anchor_array = black_points[candidate]
            best_index = candidate
            best_new_min_dists = new_min_dists
            break

    # Update the selected anchors and the min_dists.
    print(f"Improved diversity → {best_score:.4f}")
    anchors_list.append(black_points[best_index])
    anchors_list_for_plot.append(anchors_list.copy())
    if len(anchors_list) >= 2:
        del anchors_list[0]
    min_dists = np.linalg.norm(black_points - black_points[best_index], axis=1)
    # if len(anchors_list) != 1:
    #     diversity_val = np.mean(min_dists, axis=0)  # diversity is the average min distance.
    #     coverage_val = compute_coverage(anchors_list)
    #     # Overall objective: maximize diversity while penalizing coverage.
    #     best_score = -0.95 * diversity_val + 0.05 * coverage_val


for i, s in tqdm(enumerate(anchors_list_for_plot), desc="generating frames"):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Black points')
    red_point = np.mean(s, axis=0)
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    if len(s) == 1:
        ax.scatter(s[0][0],   s[0][1],   c='red', s=80, label='Red points (scaled)')
    else:
        ax.scatter(s[0][0],   s[0][1],   c='red', s=80, label='Red points (scaled)')
        ax.scatter(s[1][0],   s[1][1],   c='red', s=80, label='Red points (scaled)')
    
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
print(len(frames))
imageio.mimsave(output_gif, frames, fps=fps, loop=0)

# === Cleanup frames directory (optional) ===
for fp in frame_paths:
    os.remove(fp)
os.rmdir(frames_dir)

print(f"GIF written to {output_gif}")