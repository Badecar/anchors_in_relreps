import os
from utils import set_random_seeds
from models import Autoencoder
import numpy as np
import torch
import random
from data import load_mnist_data
from utils import set_random_seeds

# Train AE
def train_AE(num_epochs=5, batch_size=256, lr=1e-3, device='cuda', latent_dim = 2, hidden_layer = 128, trials=1, save=False, verbose=False):
    """
    Orchestrates the autoencoder pipeline:
      1. Load data
      2. Initialize the autoencoder
      3. Train and evaluate
      4. Extract embeddings
    
    Args:
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
    embeddings_list = []
    indices_list = []
    labels_list = []
    AE_list = []

    # Create the directory to save embeddings if needed.
    if save:
        save_dir = os.path.join("models", "saved_embeddings", f"dim{latent_dim}")
        os.makedirs(save_dir, exist_ok=True)

    for i in range(trials):
        set_random_seeds(i+1)
        print(f"Trial {i+1} of {trials}")
        # Create the data loaders
        train_loader, test_loader = load_mnist_data(batch_size=batch_size)
        # Initialize and train the autoencoder
        AE = Autoencoder(latent_dim=latent_dim, hidden_size=hidden_layer)
        AE.to(device)
        _, _ = AE.fit(train_loader, test_loader, num_epochs, lr, device=device, verbose=verbose)

        # Extract latent embeddings from the test loader
        embeddings, indices, labels = AE.get_latent_embeddings(test_loader, device=device)
        embeddings_list.append(embeddings.cpu().numpy())
        indices_list.append(indices.cpu())
        labels_list.append(labels.cpu().numpy())
        AE_list.append(AE)
        # Save embeddings, indices, and labels if flag is set.
        
        if save:
            np.save(os.path.join(save_dir, f'embeddings_trial_{i+1}_dim{latent_dim}.npy'), embeddings.cpu().numpy())
            np.save(os.path.join(save_dir, f'indices_trial_{i+1}_dim{latent_dim}.npy'), indices.cpu().numpy())
            np.save(os.path.join(save_dir, f'labels_trial_{i+1}_dim{latent_dim}.npy'), labels.cpu().numpy())

    return AE_list, embeddings_list, indices_list, labels_list

def load_saved_embeddings(trials=1, latent_dim=int, save_dir=None):
    """
    Loads saved embeddings, indices, and labels from the saved embeddings directory.
    
    Args:
        trials (int): Number of trials (files) to load.
        save_dir (str): Directory where the saved files are located. If None, defaults to "models/saved_embeddings"
        latent_dim (int): Dimension of the AE's latent space

    Returns:
        tuple: A tuple containing:
            - embeddings_list (list of np.ndarray): Loaded embeddings from each trial.
            - indices_list (list of np.ndarray): Loaded indices from each trial.
            - labels_list (list of np.ndarray): Loaded labels from each trial.
    """
    if save_dir is None:
        save_dir = os.path.join("models", "saved_embeddings", f"dim{latent_dim}")
    
    embeddings_list = []
    indices_list = []
    labels_list = []
    
    for i in range(trials):
        emb_path = os.path.join(save_dir, f'embeddings_trial_{i+1}_dim{latent_dim}.npy')
        idx_path = os.path.join(save_dir, f'indices_trial_{i+1}_dim{latent_dim}.npy')
        lab_path = os.path.join(save_dir, f'labels_trial_{i+1}_dim{latent_dim}.npy')
        
        # Load the saved .npy files
        embeddings = np.load(emb_path)
        indices = np.load(idx_path)
        labels = np.load(lab_path)
        
        embeddings_list.append(embeddings)
        indices_list.append(indices)
        labels_list.append(labels)
        
    return embeddings_list, indices_list, labels_list
