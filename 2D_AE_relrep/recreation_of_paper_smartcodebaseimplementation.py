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
from models import Autoencoder, train_AE, load_saved_embeddings
from data import load_mnist_data
from visualization import plot_data_list, plot_3D_relreps
from anchors import select_anchors_by_id, objective_function, greedy_one_at_a_time
from relreps import compute_relative_coordinates


# For reproducibility and consistency across runs, we set a seed
set_random_seeds(42)
from anchors import select_anchors_by_id, objective_function
from relreps import compute_relative_coordinates, compare_latent_spaces
set_random_seeds(40)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data for plotting
train_loader, test_loader = load_mnist_data()

### PARAMETERS ###
load_saved = True
latent_dim = 15
anchors_setting = 'random' # random, greedy, ids
anchor_num = 15
show_anchor_images = False
plot_results = False
nr_runs = 5

# Running experiment
if load_saved:
    embeddings_list, indices_list, labels_list = load_saved_embeddings(trials=nr_runs, latent_dim=latent_dim)
    AE_list = np.zeros(nr_runs)
else:
    # Run experiment
    AE_list, embeddings_list, indices_list, labels_list = train_AE(
        num_epochs=6,
        batch_size=256,
        lr=1e-3,
        device=device,      
        latent_dim=latent_dim,
        hidden_layer=128,
        trials=nr_runs,
        save=False,
        verbose=True
    )
print(f"Dimensions of embeddings_list: {len(embeddings_list)} x {len(embeddings_list[0])} x {len(embeddings_list[0][0])}")
print(f"Dimensions of indices_list: {len(indices_list)} x {len(indices_list[0])}")
print(f"Dimensions of labels_list: {len(labels_list)} x {len(labels_list[0])}")

# Find anchors and compute relative coordinates
manual_anchor_ids = [101, 205]
greedy_anchor_ids = greedy_one_at_a_time(embeddings_list[0], indices_list[0], 10)
random_anchor_ids = random.sample(range(len(test_loader.dataset)), anchor_num)
anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, random_anchor_ids, test_loader.dataset, show=show_anchor_images, device=device)
relative_coords_list = compute_relative_coordinates(embeddings_list, anchors_list, flatten=False)


### Plotting ###
if plot_results:
    # 3D Plot (if latent_dim=3)    
    if len(relative_coords_list[0][0]) == 3:
        for relrep in range(len(relative_coords_list)):
            plot_3D_relreps(relative_coords_list[relrep], labels_list[relrep])
    # Plot encodings side by side
    plot_data_list(embeddings_list, labels_list, do_pca=True, is_relrep=False)

    # # Plot rel_reps side by side with sign alignment
    # plot_data_list(relative_coords_list, labels_list, do_pca=True, is_relrep=True)
    # anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, random_anchor_ids, test_loader.dataset, show=False, device=device)
    # relative_coords_list = compute_relative_coordinates(embeddings_list, anchors_list, flatten=False)
    # plot_data_list(relative_coords_list, labels_list, do_pca=True, is_relrep=True)


### Similarity calculations ###
mrr_matrix, mean_mrr, cos_sim_matrix, mean_cos_sim = compare_latent_spaces(relative_coords_list, indices_list, compute_mrr=False, AE_list=AE_list)
print("\nSimilarity Results:")
if mrr_matrix[0][0] is not None:
    print(f"Mean Reciprocal Rank (MRR): {mean_mrr:.4f}")
    print("MRR Matrix:")
    print(mrr_matrix)

print(f"\nMean Cosine Similarity: {mean_cos_sim:.4f}")
print("Cosine Similarity Matrix:")
np.set_printoptions(precision=2, suppress=True)
print(cos_sim_matrix)