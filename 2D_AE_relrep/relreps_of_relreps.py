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
model = AE_conv_MNIST #AEClassifier or Autoencoder
head_type = 'reconstructor'    #reconstructor or classifier
distance_measure = 'cosine'   #cosine or euclidean
latent_dim = 2         # If load_saved: Must match an existing dim
anchor_num = 2
nr_runs = 2            # If load_saved: Must be <= number of saved runs for the dim

# Train parameters
num_epochs = 4
lr = 1e-3

# Hyperparameters for anchor selection
coverage_w = 0.8
diversity_w = 0.2
exponent = 1

# Post-processing
plot_results = True
zero_shot = False
compute_mrr = False      # Only set true if you have >32GB of RAM, and very low dim
compute_similarity = True
compute_relrep_loss = False # Can only compute if not loading from save
### ###

AE_list, emb_list_train, idx_list_train, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
    model=model,
    num_epochs=num_epochs,
    batch_size=256,
    lr=lr,
    device=device,      
    latent_dim=latent_dim,
    hidden_layer=None,
    trials=nr_runs,
    save=False,
    verbose=False,
    train_loader=train_loader,
    test_loader=test_loader
)

# Getting Tets and Validation embeddings (sorted by index)
print("Getting embeddings for test and validation set")
emb_list, idx_list, labels_list = get_embeddings(loader, AE_list, device=device)


if use_small_dataset:
    # Creates a smaller dataset from the test embeddings with balanced class counts. It is sorted by index, so each trial corresponds to each other
    small_dataset_emb, small_dataset_idx, small_dataset_labels = create_smaller_dataset(
        emb_list,
        idx_list,
        labels_list,
        samples_per_class=200
    )

    emb_list, idx_list, labels_list = small_dataset_emb, small_dataset_idx, small_dataset_labels

# Find anchors and compute relative coordinates
anchor_ids = greedy_one_at_a_time_single_euclidean(emb_list, idx_list, num_anchors=2, 
                                                          repetitions=3, diversity_weight=diversity_w, 
                                                          Coverage_weight=coverage_w, verbose=False)

anchor_list = select_anchors_by_id(AE_list, emb_list, idx_list, anchor_ids)
relrep_list = compute_relative_coordinates_euclidean(emb_list, anchor_list)

anchor_ids = greedy_one_at_a_time_single_euclidean(relrep_list, idx_list, num_anchors=2, 
                                                          repetitions=3, diversity_weight=diversity_w, 
                                                          Coverage_weight=coverage_w, verbose=False)

anchor_list_2 = select_anchors_by_id(AE_list, relrep_list, idx_list, anchor_ids)
relrep_list_2 = compute_relative_coordinates_euclidean(relrep_list, anchor_list)

if plot_results:
    # Plot encodings side by side
    # is_relrep = True
    # if is_relrep:
    #     print("Plotting relrep")
    #     plot_data_list(relrep_list, labels_list, do_pca=False, is_relrep=is_relrep, anchors_list=anch_rel)
    # else:
    #     print("Plotting absolute embeddings")
    #     plot_data_list(emb_list, labels_list, do_pca=True, is_relrep=is_relrep, anchors_list=rand_anchors_list)
    plot_data_list(emb_list, labels_list, do_pca=False, is_relrep=False, anchors_list=anchor_list)
    plot_data_list(relrep_list, labels_list, do_pca=False, is_relrep=False, anchors_list=anchor_list_2)
    plot_data_list(relrep_list_2, labels_list, do_pca=False, is_relrep=False, anchors_list=None)




### Relrep similarity and loss calculations ###
if compute_similarity:
    compare_latent_spaces(relrep_list, idx_list, compute_mrr=compute_mrr, verbose=False)