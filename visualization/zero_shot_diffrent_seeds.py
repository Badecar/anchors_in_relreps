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
from zero_shot import *
from tqdm import tqdm

# For reproducibility and consistency across runs, we set a seed
seed = 43
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
loader = val_loader
use_small_dataset = False

### PARAMETERS ###
model = AE_conv #VariationalAutoencoder, AEClassifier, or Autoencoder
head_type = 'reconstructor'    #reconstructor or classifier
distance_measure = 'cosine'   #cosine or euclidean
load_saved = False       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = False        # Save embeddings from current run
dim = 2         # If load_saved: Must match an existing dim
anchor_num = 2
nr_runs = 2            # If load_saved: Must be <= number of saved runs for the dim

# Train parameters
num_epochs = 2
lr = 1e-3

# Hyperparameters for anchor selection
coverage_w = 0.92
diversity_w = 0.08
exponent = 1

# Post-processing
zero_shot = False
plot_results = True
compute_mrr = False      # Only set true if you have >32GB of RAM, and very low dim
compute_similarity = True
compute_relrep_loss = False # Can only compute if not loading from save
### ###
# Run experiment. Return the Train embeddings
model_list, emb_list_train, idx_list_train, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
    model=model,
    num_epochs=num_epochs,
    batch_size=256,
    lr=lr,
    device=device,      
    latent_dim=dim,
    hidden_layer=None, # Select none for conv models
    nr_runs=nr_runs,
    save=save_run,
    verbose=True,
    train_loader=train_loader,
    test_loader=test_loader,
    input_dim=28*28
)

emb_list, idx_list, labels_list = get_embeddings(loader, model_list, device=device)

# Find anchors and compute relative coordinates
random_anchor_ids = random.sample(list(idx_list[0]), anchor_num)
rand_anchors_list = select_anchors_by_id(model_list, emb_list, idx_list, random_anchor_ids, loader.dataset, show=False, device=device)

# TODO: Instead of softmax, then pass the size of the weights of P into the loss. Average of the sum over each column (A)
# Optimize anchors and compute P_anchors_list
_, P_anchors_list = get_optimized_anchors(
    emb = emb_list,
    anchor_num=anchor_num,
    epochs=50,
    lr=1e-1,
    coverage_weight=coverage_w,
    diversity_weight=diversity_w,
    exponent=exponent,
    verbose=False,
    device=device
)

anch_list = rand_anchors_list
relrep_list = compute_relative_coordinates_euclidean(emb_list, anch_list)

print("Performing zero-shot stitching")
# 1. Regular autoencoder validation on the first AE
mse_reg, mse_std_reg = model_list[0].validate(loader, device=device)
print("Regular AE Validation MSE: {:.5f} ± {:.5f}".format(mse_reg, mse_std_reg))

# 2. Train the relative decoder using the first AE (freeze its encoder)
rel_decoder = train_rel_head(
    anchor_num=anchor_num,
    anchors=anch_list[0],
    num_epochs=num_epochs,
    AE=model_list[0],
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
# TODO: Are they sorted properly?

# Prepare the ground truth images from the validation set (if not already available)
all_val_images = []

for x, _ in loader:
    all_val_images.append(x.to(device))

all_val_images = torch.cat(all_val_images, dim=0)
all_val_images_flat = all_val_images.view(all_val_images.size(0), -1)

# Validate the relative decoder using the computed relative coordinates for each AE
rel_coords_tensor = torch.tensor(relrep_list[1], dtype=torch.float32)
mse_mean, mse_std = validate_relhead(
    rel_decoder,
    rel_coords_tensor,
    all_val_images_flat,
    device=device,
    show=False)
### ###
