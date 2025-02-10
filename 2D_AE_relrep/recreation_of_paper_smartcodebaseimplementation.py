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
from models import Autoencoder, train_AE
from data import load_mnist_data
from visualization import plot_data_list, plot_3D_relreps
from anchors import select_anchors_by_id, objective_function, greedy_one_at_a_time
from relreps import compute_relative_coordinates


# For reproducibility and consistency across runs, we set a seed
set_random_seeds(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.cuda.get_device_name(0))

# Run experiment
AE_list, embeddings_list, indices_list, labels_list = train_AE(
    num_epochs=5,
    batch_size=256,
    lr=1e-3,
    device=device,      
    latent_dim=10,
    hidden_layer=128,
    trials=4
)

# Find anchors and compute relative coordinates
train_loader, test_loader = load_mnist_data()
predefined_anchor_ids = [101, 205]
random_anchor_ids = random.sample(range(len(test_loader.dataset)), 10)
greedy_anchor_ids = greedy_one_at_a_time(embeddings_list[0], indices_list[0], 10)
anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, greedy_anchor_ids, test_loader.dataset, show=False, device=device)
relative_coords_list = compute_relative_coordinates(embeddings_list, anchors_list, flatten=False)


### Plotting ###

# 3D Plot (if latent_dim=3)    
if len(relative_coords_list[0][0]) == 3:
    for relrep in range(len(relative_coords_list)):
        plot_3D_relreps(relative_coords_list[relrep], labels_list[relrep])

# Plot encodings side by side
plot_data_list(embeddings_list, labels_list, do_pca=True, is_relrep=False)

# Plot rel_reps side by side with sign alignment
plot_data_list(relative_coords_list, labels_list, do_pca=True, is_relrep=True)
anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, random_anchor_ids, test_loader.dataset, show=False, device=device)
relative_coords_list = compute_relative_coordinates(embeddings_list, anchors_list, flatten=False)
plot_data_list(relative_coords_list, labels_list, do_pca=True, is_relrep=True)