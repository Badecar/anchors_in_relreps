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
from models import train_AE, load_saved_emb
from models import *
import torch.nn as nn
from data import *
from visualization import *
from anchors import *
from relreps import *
from anchors import *
from relreps import *
import matplotlib.pyplot as plt
from tqdm import tqdm

# For reproducibility and consistency across runs, we set a seed
seed=42
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data for plotting
train_loader, test_loader, val_loader = load_mnist_data()

### PARAMETERS ###
load_saved = True       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = True        # Save embeddings from current run
model = AE_conv #VariationalAutoencoder, AEClassifier, or Autoencoder, AE_conv
dim = 2         # If load_saved: Must match an existing dim
latent_dim = 100         
anchor_num = 100
nr_runs = 50
num_epochs = 10
hidden_layer = (32, 64) # (32, 64) or 128
data = "MNIST" # "FMNIST" or "MNIST". NOTE: Remember to replace load function above

# Hyperparameters for anchor selection
coverage_w = 1
diversity_w = 10
exponent = 1


if load_saved:
    emb_list, idx_list_train, labels_list_train = load_saved_emb(model, nr_runs=nr_runs, latent_dim=dim, data=data)
else:
    # Run experiment. Return the Train embeddings
    model_list, emb_list, idx_list_train, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
        model=model,
        num_epochs=10,
        batch_size=256,
        lr=1e-3,
        device=device,      
        latent_dim=2,
        nr_runs=nr_runs,
        save=save_run,
        verbose=True,
        train_loader=train_loader,
        test_loader=test_loader,
        hidden_layer=hidden_layer,
        input_dim=28*28,
        data=data
    )

_, anchors_list = get_optimized_anchors(
        emb = emb_list,
        anchor_num=2,
        epochs=100,
        lr=1e-2,
        coverage_weight=0.95,
        diversity_weight=0.05,
        exponent=1,
        dist_measure="euclidean", ## "euclidean", "mahalanobis", "cosine"
        verbose=False,
        device=device,
    )
relrep_list = compute_relative_coordinates_mahalanobis(emb_list, anchors_list)
relrep_list = np.transpose(np.array(relrep_list), (1, 0, 2))
emb_list = np.transpose(np.array(emb_list), (1, 0, 2))


norm_arr = np.mean(np.linalg.norm(emb_list, axis=2), axis=1)

eps = 1e-8                               # avoid divide‑by‑zero
lengths = np.linalg.norm(relrep_list, axis=-1, keepdims=True)
relrep_unit = relrep_list / (lengths + eps)   # shape: (N, R, 2)
std_arr = np.linalg.norm(np.std(relrep_unit, axis=1, ddof=0), axis=1)

# ------------------------------------------------------------------
#  PLOT: point cloud  +  binned mean curve  (+ error band)
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- parameters you can tweak -------------------
n_bins      = 20     # number of x‑axis slices
show_error  = False   # False → hide the shaded error band
error_type  = 'std'  # 'std'  (±1σ)   |   'sem'  (±standard error)
point_alpha = 0.10   # transparency of the raw scatter backdrop
point_size  = 8      # marker size of the raw scatter backdrop
# -------------------------------------------------------------------

x = np.asarray(norm_arr)
y = np.asarray(std_arr)

assert x.shape == y.shape, f"Shape mismatch: x {x.shape}, y {y.shape}"

# 1. build bin edges and map every x to a bin index
bins       = np.linspace(x.min(), x.max(), n_bins + 1)
bin_index  = np.digitize(x, bins) - 1           # 0 … n_bins-1
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# 2. aggregate mean (and dispersion) per bin
mean_in_bin = np.empty(n_bins)
err_in_bin  = np.empty(n_bins)

for b in range(n_bins):
    mask = bin_index == b
    if np.any(mask):
        vals = y[mask]
        mean_in_bin[b] = vals.mean()
        if show_error:
            if error_type.lower() == 'sem':
                err_in_bin[b] = vals.std(ddof=0) / np.sqrt(len(vals))
            else:                                      # population σ
                err_in_bin[b] = vals.std(ddof=0)
    else:   # empty bin → set NaN so it won’t be plotted
        mean_in_bin[b] = np.nan
        err_in_bin[b]  = np.nan

# 3. plotting
plt.figure(figsize=(7, 6))

# raw scatter for context (faint)
plt.scatter(x, y,
            s=point_size,
            alpha=point_alpha,
            edgecolors='none',
            label='Individual samples')

# mean curve
plt.plot(bin_centers, mean_in_bin,
         lw=2.5, color='C1', label='Bin mean')

# error band
if show_error:
    plt.fill_between(bin_centers,
                     mean_in_bin - err_in_bin,
                     mean_in_bin + err_in_bin,
                     color='C1', alpha=0.25,
                     label=f'±1 {error_type.upper()}')

plt.xlabel('norm_arr  (mean distance to origin)')
plt.ylabel('std_arr   (spread across runs, normalised)')
plt.title('Average variability vs. mean norm')
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()