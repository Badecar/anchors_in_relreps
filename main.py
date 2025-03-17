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
from zero_shot import *

# For reproducibility and consistency across runs, we set a seed
seed = 42
set_random_seeds(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
loader = val_loader
use_small_dataset = False

### PARAMETERS ###
model = AE_conv_MNIST #VariationalAutoencoder, AEClassifier, or Autoencoder
head_type = 'reconstructor'    #reconstructor or classifier
distance_measure = 'cosine'   #cosine or euclidean
load_saved = True       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = True        # Save embeddings from current run
dim = 30         # If load_saved: Must match an existing dim
anchor_num = dim
nr_runs = 3            # If load_saved: Must be <= number of saved runs for the dim

# Train parameters
num_epochs = 10
lr = 1e-3
hidden_layer = 128 if model != AE_conv_MNIST else None

# Hyperparameters for anchor selection
coverage_w = 0.92 # Coverage of embeddings
diversity_w = 1 - coverage_w # Pairwise between anchors
exponent = 1

# Post-processing
zero_shot = True
plot_embeddings = False
compute_mrr = False      # Only set true if you have >32GB of RAM, and very low dim
compute_similarity = True
compute_relrep_loss = False # Can only compute if not loading from save
### ###

if load_saved:
    emb_list_train, idx_list_train, labels_list_train = load_saved_emb(model, nr_runs=nr_runs, latent_dim=dim)
    model_list = load_AE_models(model=model, nr_runs=nr_runs, latent_dim=dim, hidden_layer=hidden_layer, device=device)
    # Initializing empty lists as to not break the code below
    acc_list, train_loss_list_AE,test_loss_list_AE = np.zeros(nr_runs), np.zeros(nr_runs), np.zeros(nr_runs)
else:
    # Run experiment. Return the Train embeddings
    model_list, emb_list_train, idx_list_train, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
        model=model,
        num_epochs=num_epochs,
        batch_size=256,
        lr=lr,
        device=device,      
        latent_dim=dim,
        hidden_layer=hidden_layer,
        nr_runs=nr_runs,
        save=save_run,
        verbose=True,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=28*28,
        beta=1
    )

# Getting Tets and Validation embeddings (sorted by index)
print("Getting embeddings for test and validation set")
emb_list, idx_list, labels_list = get_embeddings(loader, model_list, device=device)

# Creates a smaller dataset from the test embeddings with balanced class counts. It is sorted by index, so each run corresponds to each other
small_dataset_emb, small_dataset_idx, small_dataset_labels = create_smaller_dataset(
    emb_list,
    idx_list,
    labels_list,
    samples_per_class=400
)
if use_small_dataset: emb_list, idx_list, labels_list = small_dataset_emb, small_dataset_idx, small_dataset_labels

# Find anchors and compute relative coordinates
random_anchor_ids = random.sample(list(idx_list[0]), anchor_num)
rand_anchors_list = select_anchors_by_id(model_list, emb_list, idx_list, random_anchor_ids, loader.dataset, show=False, device=device)

# TODO: Instead of softmax, then pass the size of the weights of P into the loss. Average of the sum over each column (A)
# Optimize anchors and compute P_anchors_list
_, P_anchors_list = get_optimized_anchors(
    emb = small_dataset_emb,
    anchor_num=anchor_num,
    epochs=200,
    lr=1e-1,
    coverage_weight=coverage_w,
    diversity_weight=diversity_w,
    exponent=exponent,
    verbose=False,
    device=device,
)
anch_list = rand_anchors_list

# Compute relative coordinates for the embeddings
relrep_list = compute_relative_coordinates_euclidean(emb_list, anch_list)

### ZERO-SHOT STITCHING ###
if zero_shot:
    rel_decoder, rel_train_loss, rel_test_loss = train_rel_decoder(
        rel_model=rel_AE_conv_MNIST,
        model_list=model_list,
        relrep_list=relrep_list,
        idx_list=idx_list,
        loader=loader,
        nr_runs=nr_runs,
        device=device,
        show=True,
        verbose=True
    )

### Plotting embeddings ###
if plot_embeddings:
    do_pca = dim > 2
    print("Plotting absolute embeddings")
    plot_data_list(emb_list, labels_list, do_pca=do_pca, is_relrep=False, anchors_list=anch_list)
    print("Plotting relrep")
    plot_data_list(relrep_list, labels_list, do_pca=do_pca, is_relrep=True)

# NOTE: Watch out with comparing cosine sim between cossim- and eucl relreps.
#   The eucl relreps are only in the first quadrant, so cosine sim will be higher
### Relrep similarity and loss calculations ###
if compute_similarity:
    compare_latent_spaces(relrep_list, small_dataset_idx, compute_mrr=compute_mrr, verbose=False)