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
from models import *
import torch.nn as nn
from data import load_mnist_data
from visualization import *
from anchors import *
from relreps import *
from anchors import *
from relreps import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# For reproducibility and consistency across runs, we set a seed
seed=1
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data for plotting
train_loader, test_loader, val_loader = load_mnist_data()

### PARAMETERS ###
load_saved_emb = True       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = False        # Save embeddings from current run
latent_dim = 2         
anchor_num = 2
repetitions = 5
nr_runs = 3
num_epochs = 10
anchor_algo = "greedy" # can be "random", "greedy", "p"

# Hyperparameters for anchor selection
coverage_w = 35
diversity_w = 1
exponent = 1

abs = []
labels_list = []
anchor_ids = []
indices_list =[]
for i in tqdm(range(nr_runs), desc="creating embeddings"):
    set_random_seeds(seed+i)
    AE_list, embeddings_list, indices, labels, train_loss, test_loss, acc_list = train_AE(
        model=Autoencoder,
        num_epochs=num_epochs,
        batch_size=256,
        lr=1e-3,
        device=device,      
        latent_dim=latent_dim,
        hidden_layer=512,
        trials=1,
        save=save_run,
        verbose=False,
        train_loader=train_loader,
        test_loader=test_loader
    )
    indices_list.append(indices[0])
    labels_list.append(labels[0])
    abs.append(embeddings_list[0])

    # finding anchors depending on the chosen method
    if anchor_algo == "random":
        anchor_ids.append(random.sample(range(len(test_loader.dataset)), 2))

if anchor_algo == "greedy":
    for i in tqdm(range(nr_runs), desc="choosing anchors"):
        set_random_seeds(seed=seed+i)
        anchor_ids.append(greedy_one_at_a_time_single_euclidean(abs, indices_list,
                                                                num_anchors=anchor_num, repetitions=repetitions,
                                                                diversity_weight=diversity_w, Coverage_weight=coverage_w, verbose=False))
elif anchor_algo == "p":
    raise NotImplementedError()

else:
    raise NameError("anchor_algo has unrecognized name")
    
embeddings_list = [np.array(random.sample(embeddings.tolist(), 1000)) for embeddings in embeddings_list]
anchors = []

 
for embedding in tqdm(abs, desc="getting anchors"):
    row = []
    for ids in anchor_ids:
        temp_anchor = select_anchors_by_id(AE_list, [embedding], indices_list, ids, test_loader.dataset, show=False, device=device)[0]
        row.append(temp_anchor)
    anchors.append(row)

# print(anchor_ids)

# print(anchors)
# quit()
rel_reps = []
for embeddings, anchors_row in tqdm(zip(abs, anchors), desc="computing rel reps"):
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
    figsize=(3 * num_cols, 3 * num_rows),
    squeeze=False
)

if anchor_algo == "random":
    title = "Anchors chosen by at random and their relative representation on different seeds"
if anchor_algo == "greedy":
    title = "Anchors chosen by greedy algorithm and their relative representation on different seeds"
if anchor_algo == "p":
    raise NotImplementedError()

fig.suptitle(title, fontsize=18)

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
    for marker_idx, anchor in enumerate(anchors[i]):
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
    ax_abs.set_ylabel(f"Seed={seed+i}", fontsize=15)


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
        if i == 0:
            ax_rel.set_title(f"From anchor set {j+1}", fontsize=15)
plt.tight_layout()
plt.show()
