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
plot_results = True
compute_mrr = False      # Only set true if you have >32GB of RAM, and very low dim
compute_similarity = True
compute_relrep_loss = False # Can only compute if not loading from save
### ###

if load_saved:
    emb_list_train, idx_list_train, labels_list_train = load_saved_emb(model, trials=nr_runs, latent_dim=dim)
    model_list = load_AE_models(model=model, trials=nr_runs, latent_dim=dim, hidden_layer=hidden_layer, device=device)
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
        trials=nr_runs,
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

# Creates a smaller dataset from the test embeddings with balanced class counts. It is sorted by index, so each trial corresponds to each other
small_dataset_emb, small_dataset_idx, small_dataset_labels = create_smaller_dataset(
    emb_list,
    idx_list,
    labels_list,
    samples_per_class=400
)

if use_small_dataset:
    emb_list, idx_list, labels_list = small_dataset_emb, small_dataset_idx, small_dataset_labels

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
    device=device
)
anch_list = rand_anchors_list

# Compute relative coordinates for the embeddings
relrep_list = compute_relative_coordinates_euclidean(emb_list, anch_list)

## TODO: CURRENTLY COMPUTING RELREPS WITH THE WRONG FUNCTION (same functionality)
## TODO: PROBABLY EASIER TO PASS THE RELREPS IN INSTEAD OF THE LOADERS
## TODO: THE EMBEDDING SPACES ARE SCALED DIFFERENTLY. WE PROBABLY NEED A NORMALIZER
## TODO: note: the first zero shot seems to be working (however, it is trained on that same AE, so it doesn't really count). It would be interesting to see visually where it is sampling the other zero shots from
## TODO: THE train_rel_head FUNCTION IS PROBABLY NOT IDEAL. CAN WE PASS THE RELREPS INSTEAD OF THE LOADERS?
### ZERO-SHOT STITCHING ###
if zero_shot:
    print("Performing zero-shot stitching")
    # 1. Regular autoencoder validation on the first AE
    mse_reg, mse_std_reg = model_list[0].validate(loader, device=device)
    print("Regular AE Validation MSE: {:.5f} ± {:.5f}".format(mse_reg, mse_std_reg))

    # 2. Train the relative decoder using the relative coordinates and the validation set
    # Point 2: Train the relative decoder using the relative coordinates and the validation set targets

    from torch.utils.data import TensorDataset, DataLoader
    # New approach: use the dataset order, which is assumed to match relrep_list order.
    # Ensure relative_reps is a tensor.
    if not isinstance(relrep_list[0], torch.Tensor):
        first_relrep = torch.tensor(relrep_list[0])
    else:
        first_relrep = relrep_list[0].cpu()

    # Build target_images using the dataset’s order.
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    target_images = []
    for i in range(len(loader.dataset)):
        img, _ = loader.dataset[i]
        # In case img is not already a tensor, convert it.
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)
        img_flat = img.view(-1)
        target_images.append(img_flat)
    target_images = torch.stack(target_images, dim=0)

    # Ensure first_relrep is a tensor.
    if not isinstance(first_relrep, torch.Tensor):
        first_relrep_tensor = torch.tensor(first_relrep).clone().detach()
    else:
        first_relrep_tensor = first_relrep.cpu()  # or first_relrep.clone().detach()

    # Ensure target_images is a tensor.
    if not isinstance(target_images, torch.Tensor):
        target_images_tensor = torch.stack(target_images, dim=0)
    else:
        target_images_tensor = target_images

    # Create the TensorDataset using the proper tensors.
    rel_decoder_dataset = TensorDataset(first_relrep_tensor, target_images_tensor)
    rel_decoder_loader = DataLoader(rel_decoder_dataset, batch_size=256, shuffle=True)

    # Instantiate the zero-shot relative decoder.
    # We use the relative representation dimension from relative_reps,
    # the expected encoder output shape and number of channels from the first AE.
    rel_decoder = rel_AE_conv_MNIST(
        relative_output_dim=first_relrep.size(1),
        encoder_out_shape=model_list[0].encoder_out_shape,  # expected conv feature map shape
        n_channels=model_list[0].image_shape[0]             # e.g., 1 for MNIST
    )

    # Optionally, move the model to the device.
    rel_decoder.to(device)

    # Train the relative decoder.
    num_epochs_rel = 10  # adjust as needed
    train_losses, val_losses = rel_decoder.fit(
        train_loader=rel_decoder_loader,
        test_loader=rel_decoder_loader,
        num_epochs=num_epochs_rel,
        lr=1e-3,
        device=device,
        verbose=True
    )

    
    # Choose 3 random indices from the dataset
    sample_indices = random.sample(range(len(loader.dataset)), 3)
    # Prepare a transform to ensure images are tensors
    to_tensor = transforms.ToTensor()
    # In the plotting loop after decoding:
    for trial, _ in enumerate(model_list):
        rel_reps = relrep_list[trial]
        decoded_images = []
        original_images = []

        for idx in sample_indices:
            if not isinstance(rel_reps[idx], torch.Tensor):
                r = torch.tensor(rel_reps[idx]).unsqueeze(0).to(device)
            else:
                r = rel_reps[idx].unsqueeze(0).to(device)
            decoded = rel_decoder(r)
            decoded_img = decoded.cpu().detach().view(28, 28)
            decoded_images.append(decoded_img)

            img, _ = loader.dataset[idx]
            if not isinstance(img, torch.Tensor):
                img = to_tensor(img)
            original_img = img.squeeze()
            # If image is flattened (i.e. 784 elements), reshape it:
            if original_img.ndim == 1 and original_img.numel() == 784:
                original_img = original_img.view(28, 28)
            original_images.append(original_img)

        plt.figure(figsize=(9, 4))
        for i in range(3):
            plt.subplot(2, 3, i + 1)
            plt.imshow(original_images[i], cmap='gray')
            plt.title("Original")
            plt.axis('off')

            plt.subplot(2, 3, i + 4)
            plt.imshow(decoded_images[i], cmap='gray')
            plt.title("Decoded")
            plt.axis('off')
        plt.suptitle(f"AE Trial {trial+1}: Original vs Decoded Images")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()



### Plotting ###
if plot_results:
    do_pca = dim > 2
    
    print("Plotting absolute embeddings")
    plot_data_list(emb_list, labels_list, do_pca=do_pca, is_relrep=False, anchors_list=anch_list)
    print("Plotting relrep")
    plot_data_list(relrep_list, labels_list, do_pca=do_pca, is_relrep=True)

    # print("Plotting reconstructions")
    # visualize_reconstruction_by_id(idx_list[0][20], model_list[0], loader, device=device)

# NOTE: Watch out with comparing cosine sim between cossim- and eucl relreps.
#   The eucl relreps are only in the first quadrant, so cosine sim will be higher
### Relrep similarity and loss calculations ###
if compute_similarity:
    compare_latent_spaces(relrep_list, small_dataset_idx, compute_mrr=compute_mrr, AE_list=AE_list, verbose=False)