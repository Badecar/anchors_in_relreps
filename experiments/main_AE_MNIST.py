import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
print(f"Parent path: {parent_path}")
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from utils import *
import torch
import random
import numpy as np
from models import *
from data import *
from visualization import *
from anchors import *
from relreps import *
from zero_shot import *
import numpy as np

set_random_seeds(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_fashion_mnist_data()
data = "FMNIST" # "FMNIST" or "MNIST". NOTE: Remember to replace load function above
loader = val_loader
use_small_dataset = False # Must be false if zero-shot

### PARAMETERS ###
model = AE_conv #VariationalAutoencoder, AEClassifier, or Autoencoder, AE_conv
load_saved = True       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = True        # Save embeddings from current run
dim = 32         # If load_saved: Must match an existing dim
anchor_num = 50
nr_runs = 5            # If load_saved: Must be <= number of saved runs for the dim
hidden_layer = (32, 64) # (32, 64) or 128

# Hyperparameters for anchor selection
coverage_w = 0.9 # Coverage of embeddings
diversity_w = 1 - coverage_w # Pairwise between anchors
anti_collapse_w = 0
exponent = 1

# Post-processing
zero_shot = True
plot_embeddings = True
compute_mrr = True      # Only set true if you have >32GB of RAM
compute_similarity = True
### ###

if load_saved:
    emb_list_train, idx_list_train, labels_list_train = load_saved_emb(model, nr_runs=nr_runs, latent_dim=dim, data=data)
    model_list = load_AE_models(model=model, nr_runs=nr_runs, latent_dim=dim, hidden_layer=hidden_layer, device=device, data=data)
    # Initializing empty lists as to not break the code below
    acc_list, train_loss_list_AE,test_loss_list_AE = np.zeros(nr_runs), np.zeros(nr_runs), np.zeros(nr_runs)
else:
    # Run experiment. Return the Train embeddings
    model_list, emb_list_train, idx_list_train, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
        model=model,
        num_epochs=10,
        batch_size=256,
        lr=1e-3,
        device=device,      
        latent_dim=dim,
        hidden_layer=hidden_layer,
        nr_runs=nr_runs,
        save=save_run,
        verbose=True,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=28*28,
        data=data
    )

# Getting Tets and Validation embeddings (sorted by index)
emb_list, idx_list, labels_list = get_embeddings(loader, model_list, device=device)

# Creates a smaller dataset from the test embeddings with balanced class counts. It is sorted by index, so each run corresponds to each other
small_dataset_emb, small_dataset_idx, small_dataset_labels = create_smaller_dataset(
    emb_list,
    idx_list,
    labels_list,
    samples_per_class=200
)
if use_small_dataset: emb_list, idx_list, labels_list = small_dataset_emb, small_dataset_idx, small_dataset_labels

# Find anchors and compute relative coordinates
random_anchor_ids = random.sample(list(idx_list[0]), anchor_num)
rand_anchors_list = select_anchors_by_id(model_list, emb_list, idx_list, random_anchor_ids, loader.dataset, show=False, device=device)

# Optimize anchors and compute P_anchors_list
_, P_anchors_list, _ = get_P_anchors(
    emb = emb_list,
    anchor_num=anchor_num,
    epochs=200,
    lr=1e-3,
    coverage_weight=coverage_w,
    diversity_weight=diversity_w,
    anti_collapse_w=anti_collapse_w,
    exponent=exponent,
    dist_measure="euclidean", ## "euclidean", "mahalanobis", "cosine"
    verbose=False,
    device=device,
)

kmeans_anchors_list, _ = get_kmeans_anchors(embeddings=emb_list, anchor_num=anchor_num, idx_list=idx_list, n_closest=100, kmeans_seed=42, verbose=False)

anch_list = rand_anchors_list # P_anchors_list, rand_anchors_list, kmeans_anchors_list

# Compute relative coordinates for the embeddings
# relrep_list = compute_relative_coordinates_mahalanobis(emb_list, anch_list)
relrep_list = compute_relative_coordinates_cossim(emb_list, anch_list, flatten=False)

### ZERO-SHOT STITCHING ###
# NOTE: Decoder seems to work fine, but the relreps are hindering the performance
if zero_shot:
    rel_decoder, rel_train_loss, rel_test_loss = train_rel_decoder(
        epochs=30,
        hidden_dims=hidden_layer,
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
    if do_pca:
        title = f"PCA, {dim}D, "
    else:
        title = ""
    print("Plotting absolute embeddings")
    plot_data_list(emb_list, labels_list, do_pca=do_pca, is_relrep=True, anchors_list=anch_list, title=f"{title}Absolute Embeddings", output_file=f"embeddings_abs.png")
    print("Plotting relrep")
    plot_data_list(relrep_list, labels_list, do_pca=do_pca, is_relrep=True, title=f"{title}Relative Representations", output_file=f"embeddings_relrep.png")

# NOTE: Watch out with comparing cosine sim between cossim- and eucl relreps.
#   The eucl relreps are only in the first quadrant, so cosine sim will be higher
### Relrep similarity and loss calculations ###
if compute_similarity:
    compare_latent_spaces(relrep_list, idx_list, compute_mrr=compute_mrr, verbose=False)