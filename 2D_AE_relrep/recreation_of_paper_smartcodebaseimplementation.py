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
from visualization import visualize_reconstruction_from_embedding, visualize_image_by_idx, visualize_reconstruction_by_id, fit_and_align_pca, plot_data_list, plot_3D_relreps
from anchors import select_anchors_by_id, compute_relative_coordinates, objective_function


# For reproducibility and consistency across runs, we set a seed
set_random_seeds(42)
train_loader, test_loader, val_loader = load_mnist_data()

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
    trials=2,
    train_loader=train_loader,
    test_loader=test_loader,
)

# Find anchors and compute relative coordinates
predefined_anchor_ids = [101, 205]
random_anchor_ids = random.sample(range(len(test_loader.dataset)), 10)
anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, random_anchor_ids, test_loader.dataset, show=False, device=device)
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