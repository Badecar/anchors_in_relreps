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
from data import *
from visualization import *
from anchors import *
from relreps import *
from tqdm import tqdm

# For reproducibility and consistency across runs, we set a seed
seed = 42
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
loader = val_loader
use_small_dataset = False
### MAKE SURE THAT WE ARE USING THE SAME DATA FOR ALL FUNCTIONS ###

# TODO: CURRENTLY USING OLD VERSION OF THE AE FOR TESTING. NEED TO UPDATE TO NEW VERSION

### PARAMETERS ###
model = Autoencoder #AEClassifier or Autoencoder
head_type = 'reconstructor'    #reconstructor or classifier
distance_measure = 'cosine'   #cosine or euclidean
load_saved = False       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = True        # Save embeddings from current run
latent_dim = 30         # If load_saved: Must match an existing dim
anchor_num = 30
nr_runs = 3            # If load_saved: Must be <= number of saved runs for the dim

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
    emb_list_train, idx_list_train, labels_list_train = load_saved_emb(model, trials=nr_runs, latent_dim=latent_dim)
    AE_list = load_AE_models(model=model, trials=nr_runs, latent_dim=latent_dim, hidden_layer=128, device=device)
    if AE_list is None:
        print("No AE model found. Initializing empty AE list.")
        AE_list = np.zeros(nr_runs) # Initializing an empty AE list to use in select_anchors_by_id
    # Initializing empty lists as to not break the code below
    acc_list, train_loss_list_AE,test_loss_list_AE = np.zeros(nr_runs), np.zeros(nr_runs), np.zeros(nr_runs)
else:
    # Run experiment. Return the Train embeddings
    AE_list, emb_list_train, idx_list_train, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
        model=model,
        num_epochs=num_epochs,
        batch_size=256,
        lr=lr,
        device=device,      
        latent_dim=latent_dim,
        hidden_layer=128,
        trials=nr_runs,
        save=save_run,
        verbose=True,
        train_loader=train_loader,
        test_loader=test_loader
    )

# Getting Tets and Validation embeddings (sorted by index)
print("Getting embeddings for test and validation set")
emb_list, idx_list, labels_list = get_embeddings(loader, AE_list, device=device)

# Creates a smaller dataset from the test embeddings with balanced class counts. It is sorted by index, so each trial corresponds to each other
small_dataset_emb, small_dataset_idx, small_dataset_labels = create_smaller_dataset(
    emb_list,
    idx_list,
    labels_list,
    samples_per_class=200
)

if use_small_dataset:
    emb_list, idx_list, labels_list = small_dataset_emb, small_dataset_idx, small_dataset_labels

# Find anchors and compute relative coordinates
random_anchor_ids = random.sample(list(idx_list[0]), anchor_num)
rand_anchors_list = select_anchors_by_id(AE_list, emb_list, idx_list, random_anchor_ids, loader.dataset, show=False, device=device)

# TODO: Instead of softmax, then pass the size of the weights of P into the loss. Average of the sum over each column (A)
# Optimize anchors and compute P_anchors_list
anchor_selector, P_anchors_list = get_optimized_anchors(
    emb = small_dataset_emb,
    anchor_num=anchor_num,
    epochs=50,
    lr=1e-1,
    coverage_weight=coverage_w,
    diversity_weight=diversity_w,
    exponent=exponent,
    verbose=False,
    device=device
)

# USING RANDOM AND COSINE FOR ZERO-SHOT STICHING TESTS
anch_list = rand_anchors_list
relrep_list = compute_relative_coordinates_cossim(emb_list, anch_list)
anch_rel = compute_relative_coordinates_cossim(rand_anchors_list, anch_list) # Computing relrep for anchors

### ZERO-SHOT STICHING ###
if zero_shot:
    print("Performing zero-shot stitching")
    # 1. Regular autoencoder validation on the first AE
    mse_reg, mse_std_reg = AE_list[0].validate(loader, device=device)
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
    # Optimize anchors using the validation embeddings from each AE
    anchor_selector_val, P_anchors_list_val = get_optimized_anchors(
        emb=emb_list,
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
    relative_coords_list_P_val = compute_relative_coordinates_euclidean(emb_list, P_anchors_list_val)
    print("Comparing cosine similarity of validation relreps")
    compare_latent_spaces(relative_coords_list_P_val, idx_list, compute_mrr=compute_mrr, AE_list=AE_list, verbose=False)
    plot_data_list(emb_list, labels_list, do_pca=True, is_relrep=False, anchors_list=P_anchors_list_val)
    plot_data_list(relative_coords_list_P_val, labels_list, do_pca=True, is_relrep=True)

    # Prepare the ground truth images from the validation set (if not already available)
    all_val_images = []
    for x, _ in loader:
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
    # is_relrep = True
    # if is_relrep:
    #     print("Plotting relrep")
    #     plot_data_list(relrep_list, labels_list, do_pca=False, is_relrep=is_relrep, anchors_list=anch_rel)
    # else:
    #     print("Plotting absolute embeddings")
    #     plot_data_list(emb_list, labels_list, do_pca=True, is_relrep=is_relrep, anchors_list=rand_anchors_list)
    print("Plotting absolute embeddings")
    plot_data_list(emb_list, labels_list, do_pca=True, is_relrep=False, anchors_list=rand_anchors_list)
    print("Plotting relrep")
    plot_data_list(relrep_list, labels_list, do_pca=True, is_relrep=True, anchors_list=anch_rel)



### Relrep similarity and loss calculations ###
if compute_similarity:
    compare_latent_spaces(relrep_list, idx_list, compute_mrr=compute_mrr, AE_list=AE_list, verbose=False)