import os
import numpy as np
from utils import set_random_seeds
from data import sort_results
import torch
from .autoencoder import Autoencoder, AEClassifier
from .VAE import VariationalAutoencoder
from .AE_conv_MNIST import AE_conv_MNIST
from .load_from_save import get_save_dir

# Train AE
def train_AE(model, num_epochs=5, batch_size=256, lr=1e-3, device='cuda', latent_dim = 2, hidden_layer = 128, trials=1, input_dim=int, beta=1.0, use_test=True, save=False, verbose=False, train_loader=None, test_loader=None):
    """
    Orchestrates the autoencoder pipeline:
      1. Load data
      2. Initialize the autoencoder
      3. Train and evaluate
      4. Extract embeddings
    
    Args:
        model (class): AE Model
        num_epochs (int): Number of training epochs.
        batch_size (int): DataLoader batch size.
        lr (float): Learning rate.
        device (str): 'cpu' or 'cuda' device.
        latent_dim (int): Dimension of the AE's latent space.
    
    Returns:
        model: Trained autoencoder.
        embeddings (Tensor): Latent embeddings from the test (or train) set.
        anchors (Tensor): (Optional) set of anchor embeddings if you implement that step here.
    """
    print("Training AE models")
    embeddings_list = []
    indices_list = []
    labels_list = []
    AE_list = []
    acc_list = []
    train_loss_list = []
    test_loss_list = []

    # Create the directory to save embeddings if needed.
    
    if save:
        save_dir_emb, save_dir_AE = get_save_dir(model, latent_dim)

    for i in range(trials):
        if verbose:
            print(f"Trial {i+1} of {trials}")
        # Create the data loaders
        # Initialize and train the autoencoder
        AE = model(input_dim=input_dim, latent_dim=latent_dim, hidden_size=hidden_layer)

        AE.to(device)
        train_loss, test_loss = AE.fit(train_loader, test_loader, num_epochs, lr, device=device, verbose=verbose)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        # Extract latent embeddings from the test loader
        embeddings, indices, labels = AE.get_latent_embeddings(train_loader, device=device)
        # Sorting based on idx
        emb = embeddings.cpu().numpy()
        idx = indices.cpu()
        lab = labels.cpu().numpy()

        embeddings_sorted, idx_sorted, labels_sorted = sort_results(emb, idx, lab)

        # Appending results
        embeddings_list.append(embeddings_sorted)
        indices_list.append(idx_sorted)
        labels_list.append(labels_sorted)
        AE_list.append(AE)
        # Save embeddings, indices, and labels if flag is set.
        if hasattr(model, "accuracy"):
            acc = AE.accuracy(test_loader, device)
            acc_list.append(acc)
            if verbose: print(f'Accuracy of the network on the test images: {acc:.2f}%')
        
        if save:
            np.save(os.path.join(save_dir_emb, f'embeddings_trial_{i+1}_dim{latent_dim}.npy'), embeddings_sorted)
            np.save(os.path.join(save_dir_emb, f'indices_trial_{i+1}_dim{latent_dim}.npy'), idx_sorted)
            np.save(os.path.join(save_dir_emb, f'labels_trial_{i+1}_dim{latent_dim}.npy'), labels_sorted)
            torch.save(AE.state_dict(), os.path.join(save_dir_AE, f'ae_trial_{i+1}_dim{latent_dim}.pth'))

    return AE_list, embeddings_list, indices_list, labels_list, train_loss, test_loss, acc_list