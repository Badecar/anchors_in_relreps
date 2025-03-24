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

# For reproducibility and consistency across runs, we set a seed
set_random_seeds(43)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
data = "MNIST"
loader = val_loader
use_small_dataset = False # Must be false if zero-shot

### PARAMETERS ###
#NOTE: Conv_old gets best results with numbers, new with fashion
model = AE_conv #VariationalAutoencoder, AEClassifier, or Autoencoder
load_saved = False       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = False        # Save embeddings from current run
dim = 32         # If load_saved: Must match an existing dim
anchor_num = dim
nr_runs = 3            # If load_saved: Must be <= number of saved runs for the dim
hidden_layer = (32, 64, 128) # Use (128, 256, 512) for 100 dim, (64, 128, 256, 512) for 20 & 50 dim

# Hyperparameters for anchor selection
coverage_w = 0.90 # Coverage of embeddings
diversity_w = 1 - coverage_w # Pairwise between anchors
exponent = 1

# Post-processing
zero_shot = True
plot_embeddings = True
compute_mrr = True      # Only set true if you have >32GB of RAM, and very low dim
compute_similarity = True
### ###

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
    samples_per_class=400
)

# Find anchors and compute relative coordinates
random_anchor_ids = random.sample(list(idx_list[0]), anchor_num)
rand_anchors_list = select_anchors_by_id(model_list, emb_list, idx_list, random_anchor_ids, loader.dataset, show=False, device=device)


_, P_anchors_list = get_optimized_anchors(
    emb = small_dataset_emb,
    anchor_num=anchor_num,
    epochs=200,
    lr=1e-1,
    coverage_weight=coverage_w,
    diversity_weight=diversity_w,
    exponent=exponent,
    verbose=True,
    device=device,
)
anch_list = rand_anchors_list

# Compute relative coordinates for the embeddings
relrep_list = compute_relative_coordinates_cossim(emb_list, anch_list)

# visualize_reconstruction_by_id(idx_list[0][1], model_list[0], loader, device)


rel_decoder, rel_train_loss, rel_test_loss = train_rel_decoder(
    epochs=15,
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
