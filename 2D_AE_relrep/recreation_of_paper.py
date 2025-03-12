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
from models import *
import torch.nn as nn
from data import load_mnist_data
from visualization import plot_data_list, plot_3D_relreps
from anchors import *
from relreps import *

# For reproducibility and consistency across runs, we set a seed
seed = 42
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader = load_mnist_data()
loader = test_loader


### PARAMETERS ###
model = AEClassifier
head_type = 'classifier'    #decoder or classifier
distance_measure = 'euclidean'   # or 'euclidean'
load_saved = False       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = True        # Save embeddings from current run
latent_dim = 2         # If load_saved: Must match an existing dim
anchor_num = 2
nr_runs = 3             # If load_saved: Must be <= number of saved runs for the dim

# Train parameters
num_epochs = 10
lr = 1e-3

# Hyperparameters for anchor selection
coverage_w = 10
diversity_w = 1
exponent = 1

# Post-processing
plot_results = True
compute_mrr = False      # Only set true if you have >32GB of RAM
compute_similarity = False
compute_relrep_loss = True
### ###

if load_saved:
    embeddings_list, indices_list, labels_list = load_saved_embeddings(trials=nr_runs, latent_dim=latent_dim)
    AE_list = np.zeros(nr_runs) # Initializing an empty AE list to use in select_anchors_by_id (needed for plotting only)
else:
    # Run experiment
    AE_list, embeddings_list, indices_list, labels_list, train_loss_AE, test_loss_AE, acc_list = train_AE(
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
        train_loader=train_loader,
        test_loader=test_loader
    )
# Find anchors and compute relative coordinates
# predefined_anchor_ids = [101, 205]
random_anchor_ids = random.sample(range(len(loader.dataset)), anchor_num)
if distance_measure == 'cosine':
    greedy_anchor_ids = greedy_one_at_a_time_single_cossim(embeddings_list[0], indices_list[0],
                                                            num_anchors=anchor_num, repetitions=2,
                                                            diversity_weight=1.00, Coverage_weight=3e+2)
else:
    greedy_anchor_ids = greedy_one_at_a_time_single_euclidean(embeddings_list, indices_list,
                                                            num_anchors=2, repetitions=2,
                                                            diversity_weight=1.00, Coverage_weight=3e+2)

anchors_list = select_anchors_by_id(AE_list, embeddings_list, indices_list, random_anchor_ids, test_loader.dataset, show=False, device=device)
# relative_coords_list = compute_relative_coordinates_euclidean(embeddings_list, anchors_list, flatten=False)
relative_coords_list = compute_relative_coordinates_euclidean(embeddings_list, anchors_list, flatten=False)


### Plotting ###
if plot_results:
    # 3D Plot (if latent_dim=3)    
    if len(relative_coords_list[0][0]) == 3:
        for relrep in range(len(relative_coords_list)):
            plot_3D_relreps(relative_coords_list[relrep], labels_list[relrep])
    # Plot encodings side by side
    plot_data_list(embeddings_list, labels_list, do_pca=False, is_relrep=False)

### Relrep similarity and loss calculations ###
if compute_similarity:
    _, _, _, _ = compare_latent_spaces(relative_coords_list, indices_list, compute_mrr=compute_mrr, AE_list=AE_list)

# if compute_relrep_loss:
#     relrep_loss(
#         anchor_num=anchor_num,
#         anchors_list=anchors_list,
#         num_epochs=num_epochs,
#         AE_list=AE_list,
#         train_loader=train_loader,
#         test_loader=test_loader,
#         device=device,
#         acc_list=acc_list,
#         train_loss_AE=train_loss_AE,
#         test_loss_AE=test_loss_AE,
#         head_type=head_type,
#         distance_measure=distance_measure,
#         lr=lr,
#         verbose=False
#     )