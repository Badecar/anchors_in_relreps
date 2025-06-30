import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from tqdm import tqdm

# === Parameters ===
np.random.seed(0)
black_points = np.random.randn(100, 2)
black_points = black_points - np.mean(black_points, axis=0)
red_points    = np.array([[ 1,  1],
                          [-1, -1]])
n_frames      = 100            # number of frames in the animation
output_gif    = 'visualization_for_presentaion/linear_comp_advantage.gif'
fps           = 20

# Create a directory for frame images
frames_dir = 'visualization_for_presentaion/gif_frames'
os.makedirs(frames_dir, exist_ok=True)
frame_paths = []

# === Animation: Scaling the red points from 0.2× to 2.0× their original distance from the origin ===
dist_mean, dist_std = [0,0], [0.12,0.12]
n = 5
t_n = 20
noise_list = [np.random.normal(dist_mean, dist_std, black_points.shape) for i in range(n)]
placements = [black_points] + [black_points + noise for noise in noise_list] + [black_points]
frames_list = []
for i in range(n+1):
    start = placements[i]
    end = placements[i+1]
    for t in np.linspace(0, 1, t_n):
        temp_frame = start*(1-t) + end*t
        frames_list.append(temp_frame)



for i, s in tqdm(enumerate(frames_list), desc="generating frames"):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s[:, 0], s[:, 1], s=20, label='datapoints')
    red_point = np.mean(s, axis=0)
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    ax.scatter(red_point[0],   red_point[1],   c='red', s=80, label='linear combination of datapoints')
    
    ax.set_aspect('equal', 'box')
    ax.legend(loc="upper left")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    
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