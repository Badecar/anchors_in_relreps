from utils import set_random_seeds
from models import Autoencoder
import numpy as np
import torch
import random
from data import load_mnist_data
from utils import set_random_seeds

# Train AE
def train_AE(num_epochs=5, batch_size=256, lr=1e-3, device='cuda', latent_dim = 2, hidden_layer = 128, trials=1):
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
        latent_dim (int): Dimension of the AE's latent space (2 for easy visualization).
    
    Returns:
        model: Trained autoencoder.
        embeddings (Tensor): Latent embeddings from the test (or train) set.
        anchors (Tensor): (Optional) set of anchor embeddings if you implement that step here.
    """
    embeddings_list = []
    indices_list = []
    labels_list = []
    AE_list = []
    for i in range(trials):
        set_random_seeds(i+1)
        print(f"Trial {i+1} of {trials}")
        # Create the data loaders
        train_loader, test_loader = load_mnist_data(batch_size=batch_size)
        # Initialize and train the autoencoder
        AE = Autoencoder(latent_dim=latent_dim, hidden_size=hidden_layer)
        AE.to(device)
        train_losses, test_losses = AE.fit(train_loader, test_loader, num_epochs, lr, device=device, verbose=False)

        # Extract latent embeddings from the test loader
        embeddings, indices, labels = AE.get_latent_embeddings(train_loader, device=device)
        embeddings_list.append(embeddings.cpu().numpy()), indices_list.append(indices.cpu()), labels_list.append(labels.cpu().numpy()), AE_list.append(AE)
    return AE_list, embeddings_list, indices_list, labels_list