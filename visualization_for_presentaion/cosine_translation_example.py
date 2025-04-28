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

# Create a directory for frame images
frames_dir = 'visualization_for_presentaion/gif_frames'
os.makedirs(frames_dir, exist_ok=True)
frame_paths = []

def rotation(black_points, red_points, angle):
    # Manually selected red points

    theta = np.deg2rad(angle)

    R = np.array([[ np.cos(theta), -np.sin(theta)],
                [ np.sin(theta),  np.cos(theta)]])

    black_points = black_points.dot(R.T)
    red_points   = red_points.dot(R.T)
    return black_points, red_points

def translation(black_points, red_points, trans):
    return black_points-trans, red_points-trans

def scale(black_points, red_points, factor):
    return black_points*factor, red_points*factor

black_points_list = []
red_points_list = []

#################################
##############scale##############
#################################

black_points_list_scale_back, red_points_list_scale_back = [], []
for i in np.arange(1, 1.5, 0.05):
    black_points_temp, red_points_temp = scale(black_points, red_points, i)
    black_points_list.append(black_points_temp), red_points_list.append(red_points_temp)
    black_points_temp, red_points_temp = scale(black_points, red_points, 2.5-i)
    black_points_list_scale_back.append(black_points_temp), red_points_list_scale_back.append(red_points_temp)
black_points_list.extend(black_points_list_scale_back), red_points_list.extend(red_points_list_scale_back)

#################################
############rotation#############
#################################
angle = 45
black_points_list_rotation_back, red_points_list_rotation_back = [], []
for i in np.linspace(0, 45, 45//2):
    black_points_temp, red_points_temp = rotation(black_points, red_points, i)
    black_points_list.append(black_points_temp), red_points_list.append(red_points_temp)
    black_points_temp, red_points_temp = rotation(black_points, red_points, 45-i)
    black_points_list_rotation_back.append(black_points_temp), red_points_list_rotation_back.append(red_points_temp)

black_points_rotated, red_points_rotated = rotation(black_points, red_points, 45)
#################################
##############scale##############
#################################

black_points_list_scale_back, red_points_list_scale_back = [], []
for i in np.arange(1, 1.5, 0.05):
    black_points_temp, red_points_temp = scale(black_points_rotated, red_points_rotated, i)
    black_points_list.append(black_points_temp), red_points_list.append(red_points_temp)
    black_points_temp, red_points_temp = scale(black_points_rotated, red_points_rotated, 2.5-i)
    black_points_list_scale_back.append(black_points_temp), red_points_list_scale_back.append(red_points_temp)
black_points_list.extend(black_points_list_scale_back), red_points_list.extend(red_points_list_scale_back)
black_points_list.extend(black_points_list_rotation_back), red_points_list.extend(red_points_list_rotation_back)

for i, (b, r) in tqdm(enumerate(zip(black_points_list, red_points_list)), desc="generating frames"):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(b[:, 0], b[:, 1], s=20, label='Black points')
    red_point = np.mean(b, axis=0)
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    
    ax.scatter(r[:, 0], r[:, 1], s=20, label='Black points')
    for x, y in r:
        plt.arrow(0, 0, x, y,
                head_width=0.05,
                head_length=0.1,
                length_includes_head=True,
                color='red')
    r_1_norm = r[0]/(np.linalg.norm(r[0]))
    r_2_norm = r[1]/(np.linalg.norm(r[1]))
    cos_sim = np.dot(r_1_norm, r_2_norm)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Cosine similarity: {cos_sim:.3f}")
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

n_frames      = 105            # number of frames in the animation
output_gif    = 'visualization_for_presentaion/cosine_sim_trans.gif'
fps           = 20

# Create a directory for frame images
frames_dir = 'visualization_for_presentaion/gif_frames'
frame_paths = []

black_points_list = []
red_points_list = []

#################################
##############trans##############
#################################

black_points_list_scale_back, red_points_list_scale_back = [], []
for x, y in zip(np.linspace(0, 0.3, 30), np.linspace(0, 0.1, 30)):
    black_points_temp, red_points_temp = translation(black_points, red_points, np.array([x,y]))
    black_points_list.append(black_points_temp), red_points_list.append(red_points_temp)

black_points_trans, red_points_trans = translation(black_points, red_points, np.array([0.3,0.1]))

for x, y in zip(np.linspace(0, 0.45, 45), np.linspace(0, 0.2, 45)):
    black_points_temp, red_points_temp = translation(black_points_trans, red_points_trans, np.array([-x,y]))
    black_points_list.append(black_points_temp), red_points_list.append(red_points_temp)

black_points_trans, red_points_trans = translation(black_points_trans, red_points_trans, np.array([-0.45,0.2]))

for x, y in zip(np.linspace(0, 0.15, 30), np.linspace(0, 0.3, 30)):
    black_points_temp, red_points_temp = translation(black_points_trans, red_points_trans, np.array([x,-y]))
    black_points_list.append(black_points_temp), red_points_list.append(red_points_temp)

for i, (b, r) in tqdm(enumerate(zip(black_points_list, red_points_list)), desc="generating frames"):
    # Apply scaling to the red points
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(b[:, 0], b[:, 1], s=20, label='Black points')
    
    ax.axhline(0, color='grey', linewidth=1)
    ax.axvline(0, color='grey', linewidth=1)
    
    ax.scatter(r[:, 0], r[:, 1], s=40, label='Black points')
    for x, y in r:
        plt.arrow(0, 0, x, y,
                head_width=0.05,
                head_length=0.1,
                length_includes_head=True,
                color='red')
    r_1_norm = r[0]/(np.linalg.norm(r[0]))
    r_2_norm = r[1]/(np.linalg.norm(r[1]))
    cos_sim = np.dot(r_1_norm, r_2_norm)
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Cosine similarity: {cos_sim:.3f}")
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
