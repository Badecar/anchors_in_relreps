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
from models import train_AE, load_saved_embeddings, train_decoder_on_relreps
from data import load_mnist_data
from visualization import plot_data_list, plot_3D_relreps
from anchors import select_anchors_by_id, objective_function, greedy_one_at_a_time, greedy_one_at_a_time_single_euclidean
from relreps import compute_relative_coordinates, compute_relative_coordinates_euclidean, compare_latent_spaces


# For reproducibility and consistency across runs, we set a seed
set_random_seeds(32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

### PARAMETERS ###
load_saved = False       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = False        # Save embeddings from current run
use_test = False
latent_dim = 2         # If load_saved: Must match an existing dim
anchor_num = 2
nr_runs = 1             # If load_saved: Must be <= number of saved runs for the dim
plot_results = False
compute_mrr = False      # Only set true if you have >32GB of RAM
compute_similarity = False
compute_relrep_loss = True

# Hyperparameters for anchor selection
coverage_w = 1
diversity_w = 0.5
exponent = 0.25

### Running experiment ###
train_loader, test_loader = load_mnist_data()
if not use_test: test_loader = train_loader

if load_saved:
    embeddings_list, indices_list, labels_list = load_saved_embeddings(trials=nr_runs, latent_dim=latent_dim)
    AE_list = np.zeros(nr_runs) # Initializing an empty AE list to use in select_anchors_by_id (needed for plotting only)
else:
    # Run experiment
    AE_list, embeddings_list, indices_list, labels_list, train_loss_AE, test_loss_AE = train_AE(
        num_epochs=6,
        batch_size=256,
        lr=1e-3,
        device=device,      
        latent_dim=latent_dim,
        hidden_layer=128,
        trials=nr_runs,
        use_test=False,
        save=save_run,
        verbose=False
    )

# Find anchors and compute relative coordinates
train_loader, test_loader = load_mnist_data()
predefined_anchor_ids = [101, 205]
random_anchor_ids = random.sample(range(len(test_loader.dataset)), 2)


greedy_anchor_ids = greedy_one_at_a_time_single_euclidean(embeddings_list[0], indices_list[0],
                                                          num_anchors=2, repetitions=2,
                                                          diversity_weight=1.00, Coverage_weight=3e+2)


anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, greedy_anchor_ids, test_loader.dataset, show=False, device=device)
relative_coords_list = compute_relative_coordinates_euclidean(embeddings_list, anchors_list, flatten=False)

### Plotting ###
if plot_results:
    # 3D Plot (if latent_dim=3)    
    if len(relative_coords_list[0][0]) == 3:
        for relrep in range(len(relative_coords_list)):
            plot_3D_relreps(relative_coords_list[relrep], labels_list[relrep])
    # Plot encodings side by side
    plot_data_list(embeddings_list, labels_list, do_pca=True, is_relrep=False)

    #Nikolaj's plotting code

    # # # Plot rel_reps side by side with sign alignment
    # plot_data_list(relative_coords_list, labels_list, do_pca=True, is_relrep=True)
    # anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, random_anchor_ids, test_loader.dataset, show=False, device=device)
    # relative_coords_list = compute_relative_coordinates(embeddings_list, anchors_list, flatten=False)
    # plot_data_list(relative_coords_list, labels_list, do_pca=True, is_relrep=True)


### Similarity calculations ###
if compute_similarity:
    mrr_matrix, mean_mrr, cos_sim_matrix, mean_cos_sim = compare_latent_spaces(relative_coords_list, indices_list, compute_mrr=compute_mrr, AE_list=AE_list)
    print("\nSimilarity Results:")
    np.set_printoptions(precision=2, suppress=True)
    if compute_mrr:
        print(f"Mean Reciprocal Rank (MRR): {mean_mrr:.4f}")
        print("MRR Matrix:")
        print(mrr_matrix)

    print(f"\nMean Cosine Similarity: {mean_cos_sim:.4f}")
    print("Cosine Similarity Matrix:")
    print(cos_sim_matrix)


if compute_relrep_loss:
    # Train decoder on relative representations for comparison
    print("\nTraining decoder on relative representations...")
    decoder_model, train_loss_dec, test_loss_dec = train_decoder_on_relreps(
        anchors=anchors_list,
        train_loader=train_loader,
        test_loader=test_loader,
        model=AE_list[0],
        num_epochs=6,
        lr=1e-3,
        device=device,
        verbose=False
    )

    # Print the full list of losses per epoch
    print("\nFull Autoencoder Training Losses per Epoch:")
    print("Train Losses:", train_loss_AE)
    print("Test Losses:", test_loss_AE)

    print("\nDecoder-Only Training Losses per Epoch (using relative representations):")
    print("Train Losses:", train_loss_dec)
    print("Test Losses:", test_loss_dec)
