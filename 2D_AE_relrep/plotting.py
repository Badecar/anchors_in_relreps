import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from utils import set_random_seeds
import torch
import random
import numpy as np
from models import train_AE, load_saved_emb
from data import load_mnist_data
from visualization import *
from anchors import *
from relreps import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# For reproducibility and consistency across runs, we set a seed
seed=42
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data for plotting
train_loader, test_loader, val_loader = load_mnist_data()

### PARAMETERS ###
load_saved_emb = True       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = False        # Save embeddings from current run
latent_dim = 2         # If load_saved: Must match an existing dim
anchor_num = 2
repetitions = 1
nr_runs = 3             # If load_saved: Must be <= number of saved runs for the dim
plot_results = True
compute_mrr = True      # Only set true if you have >32GB of RAM

# Hyperparameters for anchor selection
coverage_w = 10
diversity_w = 1
exponent = 1

abs = []
labels_list = []
anchor_random_ids = []
for i in range(nr_runs):
    set_random_seeds(seed)
    AE_list, embeddings_list, indices_list, labels = train_AE(
        num_epochs=2,
        batch_size=256,
        lr=1e-3,
        device=device,      
        latent_dim=latent_dim,
        hidden_layer=512,
        trials=1,
        save=save_run,
        verbose=False,
        train_loader=train_loader,
        test_loader=test_loader,
        seed=seed+i
    )
    labels_list.append(labels[0])
    abs.append(embeddings_list[0])
    anchor_random_ids.append(random.sample(range(len(test_loader.dataset)), 2))
embeddings_list = [np.array(random.sample(embeddings.tolist(), 1000)) for embeddings in embeddings_list]
anchors_random = []

 
for embedding in tqdm(abs, desc="getting anchors"):
    row = []
    for ids in anchor_random_ids:
        temp_anchor_random = select_anchors_by_id(AE_list, [embedding], indices_list, ids, test_loader.dataset, show=False, device=device)[0]
        row.append(temp_anchor_random)
    anchors_random.append(row)

rel_reps = []
for embeddings, anchors_row in tqdm(zip(abs, anchors_random), desc="computing rel reps"):
    temp_rel_reps_row = compute_relative_coordinates_euclidean([embeddings]*len(anchors_row), anchors_row)
    rel_reps.append(temp_rel_reps_row)





# Number of rows = number of seeds
num_rows = len(abs)
# Number of columns = 1 + however many rel_reps you have per row
num_cols = 1 + len(rel_reps[0])
# squeeze=False ensures that axes is always 2D with shape (num_rows, num_cols)
fig, axes = plt.subplots(
    nrows=num_rows,
    ncols=num_cols,
    figsize=(4 * num_cols, 4 * num_rows),
    squeeze=False
)
markers = ["v", "^", "<", ">"]
for i in tqdm(range(num_rows), desc="plotting"):
    # Plot ABS in column 0
    ax_abs = axes[i, 0]
    ax_abs.scatter(
        abs[i][:, 0],
        abs[i][:, 1],
        c=labels_list[i],
        cmap='tab10',
        alpha=0.7,
        s=10
    )
    for marker_idx, anchor in enumerate(anchors_random[i]):
        ax_abs.scatter(
            anchor[:, 0],
            anchor[:, 1],
            c="black",
            marker=markers[marker_idx],
            alpha=1,
            s=25,
            label=f"Anchor Set {marker_idx+1}"
        )
    ax_abs.legend()
    ax_abs.set_title(f"ABS[{i}]", fontsize=10)

    # Plot each of the rel_reps in columns 1..N
    for j in range(len(rel_reps[i])):
        ax_rel = axes[i, j+1]
        ax_rel.scatter(
            rel_reps[i][j][:, 0],
            rel_reps[i][j][:, 1],
            c=labels_list[i],
            cmap='tab10',
            alpha=0.7,
            s=10
        )
        ax_rel.set_title(f"REL REP[{i}][{j}]", fontsize=10)
plt.tight_layout()
plt.show()
