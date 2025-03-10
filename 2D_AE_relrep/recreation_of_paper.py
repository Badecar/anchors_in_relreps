import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from utils import *
import torch
import random
import numpy as np
from models import *
import torch.nn as nn
from data import load_mnist_data
from visualization import *
from anchors import *
from relreps import *

# For reproducibility and consistency across runs, we set a seed
seed = 43
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
loader = test_loader

### PARAMETERS ###
model = Autoencoder #or Autoencoder
head_type = 'reconstructor'    #reconstructor or classifier
distance_measure = 'euclidean'   # or 'euclidean'
load_saved = True       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = True        # Save embeddings from current run
latent_dim = 10         # If load_saved: Must match an existing dim
anchor_num = 10
nr_runs = 3             # If load_saved: Must be <= number of saved runs for the dim

# Train parameters
num_epochs = 7
lr = 1e-3

# Hyperparameters for anchor selection
coverage_w = 0.92
diversity_w = 0.08
exponent = 1

# Post-processing
plot_results = True
zero_shot = False
compute_mrr = False      # Only set true if you have >32GB of RAM, and very low dim
compute_similarity = True
compute_relrep_loss = False # Can only compute if not loading from save
### ###

if load_saved:
    embeddings_list, indices_list, labels_list = load_saved_emb(trials=nr_runs, latent_dim=latent_dim)
    AE_list = load_AE_models(model=model, trials=nr_runs, latent_dim=latent_dim, hidden_layer=128, device=device)
    if AE_list is None:
        print("No AE model found. Initializing empty AE list.")
        AE_list = np.zeros(nr_runs) # Initializing an empty AE list to use in select_anchors_by_id (needed for plotting only)
    acc_list = np.zeros(nr_runs)
    train_loss_list_AE = np.zeros(nr_runs)
    test_loss_list_AE = np.zeros(nr_runs)
else:
    # Run experiment
    AE_list, embeddings_list, indices_list, labels_list, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
        model=model,
        num_epochs=num_epochs,
        batch_size=256,
        lr=lr,
        device=device,      
        latent_dim=latent_dim,
        hidden_layer=128,
        trials=nr_runs,
        use_test=False, #Must be true if we are comparing the loss with the relrep
        save=save_run,
        verbose=False,
        seed=seed,
        train_loader=train_loader,
        test_loader=test_loader
    )

# Creates a smaller dataset from the train embeddings with balanced class counts. It is sorted by index, so each trial corresponds to each other
smaller_dataset_embeddings, smaller_dataset_indices, smaller_dataset_labels = create_smaller_dataset(
    embeddings_list,
    indices_list,
    labels_list,
    samples_per_class=200
)

# Find anchors and compute relative coordinates
# TODO: Creating the validation dataset screwed with random. Getting out of bounds IDs
# random_anchor_ids = random.sample(range(len(test_loader.dataset)), anchor_num)
# rand_anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, random_anchor_ids, test_loader.dataset, show=False, device=device)

# TODO: Instead of softmax, then pass the size of the weights of P into the loss. Average of the sum over each column (A)
# Optimize anchors and compute P_anchors_list
anchor_selector, P_anchors_list = get_optimized_anchors(
    emb = smaller_dataset_embeddings,
    anchor_num=anchor_num,
    epochs=50,
    lr=1e-1,
    coverage_weight=coverage_w,
    diversity_weight=diversity_w,
    exponent=exponent,
    verbose=False,
    device=device
)
relative_coords_list_P = compute_relative_coordinates_euclidean(smaller_dataset_embeddings, P_anchors_list, flatten=False)
# relative_coords_list_rand = compute_relative_coordinates_euclidean(smaller_dataset_embeddings, rand_anchors_list, flatten=False)

### ZERO-SHOT STICHING ###
if zero_shot:
    print("Performing zero-shot stitching")
    # 1. Regular autoencoder validation on the first AE
    mse_reg, mse_std_reg = AE_list[0].validate(val_loader, device=device)
    print("Regular AE Validation MSE: {:.5f} ± {:.5f}".format(mse_reg, mse_std_reg))

    # 2. Train the relative decoder using the first AE (freeze its encoder)
    rel_decoder = train_rel_head(
        anchor_num=anchor_num,
        anchors=P_anchors_list[0],
        num_epochs=num_epochs,
        AE=AE_list[0],
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        acc=acc_list,
        train_loss_AE=train_loss_list_AE,
        test_loss_AE=test_loss_list_AE,
        head_type=head_type,
        distance_measure=distance_measure,
        lr=lr,
        verbose=False,
        show_AE_loss=False # CANT run from save if this is True
    )
    print("Relative decoder training complete.")

    # 3. Evaluate every AE using the trained relative decoder
    # Gather latent embeddings from the validation set for each AE using AE.get_latent_embeddings()
    val_embeddings_list = []
    val_indices_list = []
    val_labels_list = []
    for ae in AE_list:
        val_embeddings, val_indices, val_labels = ae.get_latent_embeddings(val_loader, device=device)
        val_embeddings_list.append(val_embeddings.cpu().numpy())
        val_indices_list.append(val_indices.cpu().numpy())
        val_labels_list.append(val_labels.cpu().numpy())

    # Optimize anchors using the validation embeddings from each AE
    anchor_selector_val, P_anchors_list_val = get_optimized_anchors(
        emb=val_embeddings_list,
        anchor_num=anchor_num,
        epochs=50,
        lr=1e-1,
        coverage_weight=coverage_w,
        diversity_weight=diversity_w,
        exponent=exponent,
        verbose=False,
        device=device
    )

    # TODO: THEY DO NOT LOOK LIKE THEY ARE SORTED PROPERLY WHEN 
    relative_coords_list_P_val = compute_relative_coordinates_euclidean(val_embeddings_list, P_anchors_list_val)
    print("Comparing cosine similarity of validation relreps")
    compare_latent_spaces(relative_coords_list_P_val, val_indices_list, compute_mrr=compute_mrr, AE_list=AE_list, verbose=False)
    plot_data_list(val_embeddings_list, val_labels_list, do_pca=True, is_relrep=False, anchors_list=P_anchors_list_val)
    plot_data_list(relative_coords_list_P_val, val_labels_list, do_pca=True, is_relrep=True)

    # Prepare the ground truth images from the validation set (if not already available)
    all_val_images = []
    for x, _ in val_loader:
        all_val_images.append(x.to(device))
    all_val_images = torch.cat(all_val_images, dim=0)
    all_val_images_flat = all_val_images.view(all_val_images.size(0), -1)

    # Validate the relative decoder using the computed relative coordinates for each AE
    for relrep in range(len(relative_coords_list_P_val)):
        rel_coords_tensor = torch.tensor(relative_coords_list_P_val[relrep], dtype=torch.float32)
        mse_mean, mse_std = validate_relhead(
            rel_decoder,
            rel_coords_tensor,
            all_val_images_flat,
            device=device,
            show=True)
        print(f"Validation AE {relrep} relative decoder validation MSE: {mse_mean:.5f} ± {mse_std:.5f}")
    ### ###

### Plotting ###
if plot_results:
    # Plot encodings side by side
    plot_data_list(relative_coords_list_P, smaller_dataset_labels, do_pca=True, is_relrep=True, anchors_list=P_anchors_list)

### Relrep similarity and loss calculations ###
if compute_similarity:
    print("Computing similarity for P")
    compare_latent_spaces(relative_coords_list_P, smaller_dataset_indices, compute_mrr=compute_mrr, AE_list=AE_list, verbose=False)