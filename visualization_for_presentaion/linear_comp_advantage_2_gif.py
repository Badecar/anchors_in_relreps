import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm

# === Parameters ===
np.random.seed(0)
black_points = np.random.randn(100, 2)
black_points = black_points - np.mean(black_points, axis=0)
red_points    = np.array([[ 1.2,  0.4],
                          [1, -1.1]])
n_frames      = 200            # number of frames in the animation
output_gif    = 'visualization_for_presentaion/linear_comp_advantage_2.gif'
fps           = 20

# Create a directory for frame images
frames_dir = 'visualization_for_presentaion/gif_frames'
os.makedirs(frames_dir, exist_ok=True)
frame_paths = []
frames_list = []
frames_list.append([black_points, red_points])
frame_number = 0
for i in range(40):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_points[:, 0],   red_points[:, 1],   c='red', s=20, label='Basis for anchor')
    
    ax.set_aspect('equal', 'box')
    ax.legend(loc="upper left")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])

    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)
    frame_number += 1

a = np.linspace(0, 1, 20)
for i, j in enumerate(a):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_points[:, 0],   red_points[:, 1],   c='red', s=20, label='Basis for anchor')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])

    plt.arrow(0, 0, red_points[0][0], red_points[0][1],
            head_width=0.05,
            head_length=0.1,
            alpha=j,
            length_includes_head=True,
            color='red')
    plt.arrow(0, 0, red_points[1][0], red_points[1][1],
            head_width=0.05,
            head_length=0.1,
            alpha=j,
            length_includes_head=True,
            color='red')
    ax.legend(loc="upper left")
    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)
    frame_number += 1
    
for i, j in enumerate(a):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_points[:, 0],   red_points[:, 1],   c='red', s=20, label='Basis for anchor')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])

    plt.arrow(0 + red_points[1][0]*j, 0 + red_points[1][1]*j, red_points[0][0], red_points[0][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    plt.arrow(0, 0, red_points[1][0], red_points[1][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    ax.legend(loc="upper left")
    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)
    frame_number += 1
a = np.linspace(0, 1, 10)

for i, j in enumerate(a):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_points[:, 0],   red_points[:, 1],   c='red', s=20, label='Basis for anchor', alpha=1-j*0.5)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_title(f"weight 1: {1.0}, weight 2: {1.0}")
    plt.arrow(0 + red_points[1][0], 0 + red_points[1][1], red_points[0][0], red_points[0][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    plt.arrow(0, 0, red_points[1][0], red_points[1][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    ax.scatter(red_points[0, 0] + red_points[1, 0],   red_points[0, 1] + red_points[1, 1],   c='red', s=80,alpha=j, label="Anchor")
    ax.legend(loc="upper left")

    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    frame_number += 1
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)
r = np.linalg.norm(red_points[0]+red_points[1])           # radius of the circle

# angle in radians, measured from +x axis toward +y
v = red_points[0] + red_points[1]
angle_rad = np.arctan2(v[1], v[0])

n = 100
theta_vals  = np.linspace(0, 2*np.pi, n, endpoint=True) + angle_rad

# pre‑compute the inverse of the 2×2 matrix [v1 v2]
A_inv = np.linalg.inv(red_points.T)          # columns are the basis vectors

for k, theta in enumerate(theta_vals):

    # ---- target point on the circle ----
    target = r * np.array([np.cos(theta), np.sin(theta)])

    # ---- solve for weights w1, w2 so that w1*v1 + w2*v2 = target ----
    w1, w2 = A_inv @ target

    # ---- build the two component vectors ----
    vec2 = w2 * red_points[1]                # second basis vector, scaled
    vec1 = w1 * red_points[0]                # first basis vector, scaled
    tail_vec1 = vec2                         # tail of the second arrow
    head_point = tail_vec1 + vec1            # should equal `target`

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)

    # basis vectors (small red dots for reference)
    ax.scatter(red_points[:, 0], red_points[:, 1], c='red', s=20, alpha=0.5, label="Basis for anchor")

    # first component arrow (origin → w2*v2)
    ax.arrow(0, 0, vec2[0], vec2[1],
             length_includes_head=True,
             head_width=0.07, head_length=0.15,
             color='red')

    # second component arrow (tail = w2*v2 → head = target)
    ax.arrow(tail_vec1[0], tail_vec1[1],
             vec1[0], vec1[1],
             length_includes_head=True,
             head_width=0.07, head_length=0.15,
             color='red')

    # moving point on the circle
    ax.scatter(head_point[0], head_point[1], c='red', s=80, label="Anchor")

    # cosmetics
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5);  ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_title(f"weight 1: {w1:.1f}, weight 2: {w2:.1f}")
    ax.legend(loc='upper left')

    # save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    frame_number += 1
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)

for i, j in enumerate(a):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_points[:, 0],   red_points[:, 1],   c='red', s=20, label='Basis for anchor', alpha=0.5 + j*0.5)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_title(f"weight 1: {1.0}, weight 2: {1.0}")
    plt.arrow(0 + red_points[1][0], 0 + red_points[1][1], red_points[0][0], red_points[0][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    plt.arrow(0, 0, red_points[1][0], red_points[1][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    ax.scatter(red_points[0, 0] + red_points[1, 0],   red_points[0, 1] + red_points[1, 1],   c='red', s=80, alpha=1-j, label="Anchor")
    
    ax.legend(loc="upper left")

    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    frame_number += 1
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)

a = np.linspace(0, 1, 20)
for i, j in enumerate(a):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_points[:, 0],   red_points[:, 1],   c='red', s=20, label='Basis for anchor')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])

    plt.arrow(0 + red_points[1][0]*(1-j), 0 + red_points[1][1]*(1-j), red_points[0][0], red_points[0][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    plt.arrow(0, 0, red_points[1][0], red_points[1][1],
            head_width=0.05,
            head_length=0.1,
            length_includes_head=True,
            color='red')
    
    ax.legend(loc="upper left")
    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)
    frame_number += 1

for i, j in enumerate(a):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(black_points[:, 0], black_points[:, 1], s=20, label='Datapoints')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_points[:, 0],   red_points[:, 1],   c='red', s=20, label='Basis for anchor')
    
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3]);  ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])

    plt.arrow(0, 0, red_points[0][0], red_points[0][1],
            head_width=0.05,
            head_length=0.1,
            alpha=1-j,
            length_includes_head=True,
            color='red')
    plt.arrow(0, 0, red_points[1][0], red_points[1][1],
            head_width=0.05,
            head_length=0.1,
            alpha=1-j,
            length_includes_head=True,
            color='red')
    
    ax.legend(loc="upper left")

    # Save frame
    frame_path = os.path.join(frames_dir, f'frame_{frame_number:03d}.png')
    fig.savefig(frame_path, dpi=80)
    plt.close(fig)
    frame_paths.append(frame_path)
    frame_number += 1

# === Combine frames into a GIF ===
frames = [imageio.imread(fp) for fp in frame_paths]
imageio.mimsave(output_gif, frames, fps=fps, loop=0)

# === Cleanup frames directory (optional) ===
for fp in frame_paths:
    os.remove(fp)

print(f"GIF written to {output_gif}")