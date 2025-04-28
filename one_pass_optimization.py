from zero_shot import rel_AE_conv_MNIST
from models import AE_conv, load_saved_emb, load_AE_models, train_AE
from utils import *
from data import load_mnist_data
from anchors import train_relrep_decoder, one_pass_optimization, pretrain_P

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


set_random_seeds(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
data = "MNIST" # "FMNIST" or "MNIST". NOTE: Remember to replace load function above
loader = train_loader

### PARAMETERS ###
model = AE_conv #VariationalAutoencoder, AEClassifier, or Autoencoder, AE_conv
decoder = rel_AE_conv_MNIST
load_saved = False       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = True        # Save embeddings from current run
load_model = False
save_model = True
dim = 200         # If load_saved: Must match an existing dim
epoch = 10
anchor_num = dim
nr_runs = 20            # If load_saved: Must be <= number of saved runs for the dim
hidden_layer = (32, 64) # (32, 64) or 128

if load_saved:
    emb_list, idx_list, labels_list_train = load_saved_emb(model, nr_runs=nr_runs, latent_dim=dim, data=data)
    model_list = load_AE_models(model=model, nr_runs=nr_runs, latent_dim=dim, hidden_layer=hidden_layer, device=device, data=data)
    # Initializing empty lists as to not break the code below
    acc_list, train_loss_list_AE,test_loss_list_AE = np.zeros(nr_runs), np.zeros(nr_runs), np.zeros(nr_runs)
else:
    # Run experiment. Return the Train embeddings
    model_list, emb_list, idx_list, labels_list_train, train_loss_list_AE, test_loss_list_AE, acc_list = train_AE(
        model=model,
        num_epochs=10,
        batch_size=256,
        lr=1e-3,
        device=device,      
        latent_dim=dim,
        hidden_layer=hidden_layer,
        nr_runs=nr_runs,
        save=save_run,
        verbose=False,
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=28*28,
        data=data
    )



# ------------------------------------------------------------------
# Stage-0: P initial diversification
# ------------------------------------------------------------------



one_pass_optimization_model = one_pass_optimization(
        embed_banks = emb_list,
        num_anchors = anchor_num,
        device=device,
        similarity = "mahalanobis",  # or "cosine"
        row_softmax = True,  # softmax over P rows → convex combos
        )
path = "one_pass_optimization_model.pth"
ckpt_path = Path(path)   # add .pth for clarity

if ckpt_path.is_file() and load_model:
    # -- load pre-trained weights -----------------------------------------
    one_pass_optimization_model.load_model(path, map_location=device)
    print(f"✓ Loaded checkpoint from {ckpt_path}")
    one_pass_optimization_model = one_pass_optimization_model.to(device)

else:
    pretrain_P(one_pass_optimization_model,
           epochs=1000,
           lr=1e-2,
           device=device)
    
    # -- create id lookup table to speed up training -----------------------------------------------

    id_lookup = []
    for ids_vec in idx_list:                # np.array of ids for that bank
        lut = torch.full((ids_vec.max()+1,), -1, dtype=torch.long)
        lut[torch.tensor(ids_vec)] = torch.arange(len(ids_vec))
        id_lookup.append(lut.to(device))

    # -- train from scratch -----------------------------------------------
    one_pass_optimization_model = train_relrep_decoder(
        model            = one_pass_optimization_model,
        dataloader       = train_loader,
        id_lookup        = id_lookup,
        embeddings_list  = emb_list,      # make sure this matches the function signature
        epochs           = epoch,
        lr               = 5e-3,
        device           = device,
        verbose=True
    )
    if save_model:
        one_pass_optimization_model.save_model(path)
    print(f"✓ Trained new model and saved to {ckpt_path}")


show_n = 10
shown  = 0

for images, (ids, _) in train_loader:              # original loader
    img_flat = images[0].to(device)
    img_id   = ids[0].item()
    # find row in emb_list[0] whose identifier matches img_id
    idx = np.where(idx_list[0] == img_id)[0][0]
    emb = torch.tensor(emb_list[0][idx], device=device)

    one_pass_optimization_model.visualize_reconstruction(emb, img_flat)
    shown += 1
    if shown >= show_n:
        break


# ------------------------------------------------------------------
# 1.  Gather data  (embeddings, anchors, labels)
# ------------------------------------------------------------------
emb_2d  = emb_list[0]               # shape [N, 2]
labels  = labels_list_train[0]      # shape [N]  (0‒9 integers)

# convert to NumPy if they’re tensors
if isinstance(emb_2d, torch.Tensor):
    emb_2d = emb_2d.cpu().numpy()
if isinstance(labels, torch.Tensor):
    labels = labels.cpu().numpy()

anchors = one_pass_optimization_model.get_anchors().cpu().detach().numpy()  # [m, 2]

# ------------------------------------------------------------------
# 2.  Build a discrete colormap for 10 digit classes
# ------------------------------------------------------------------
cmap = ListedColormap(plt.get_cmap("tab10").colors[:10])  # tab10 has 10 distinct colors

# ------------------------------------------------------------------
# 3.  Scatter-plot
# ------------------------------------------------------------------
plt.figure(figsize=(6, 6))
sc = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                 c=labels, s=2, cmap=cmap, alpha=0.1, label="embeddings")
plt.scatter(anchors[:, 0], anchors[:, 1],
            s=140, c="red", marker="x", linewidths=2.0, label="anchors")

plt.title("2-D Embeddings colored by label with learned anchors")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend(loc="upper right")
plt.colorbar(sc, ticks=range(10), label="Digit label")
plt.tight_layout()
plt.show()