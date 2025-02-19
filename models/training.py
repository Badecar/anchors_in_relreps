import os
from utils import set_random_seeds
from models import Autoencoder
import numpy as np
import torch
import random
import torch.nn as nn
from data import load_mnist_data
from utils import set_random_seeds
import torch.nn.functional as F
from tqdm import tqdm

# Train AE
def train_AE(num_epochs=5, batch_size=256, lr=1e-3, device='cuda', latent_dim = 2, hidden_layer = 128, trials=1, use_test=True, save=False, verbose=False):
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
        train_loss, test_loss = AE.fit(train_loader, test_loader, num_epochs, lr, device=device, verbose=verbose)

        # Extract latent embeddings from the test loader
        if use_test:
            embeddings, indices, labels = AE.get_latent_embeddings(test_loader, device=device)
        else:
            embeddings, indices, labels = AE.get_latent_embeddings(train_loader, device=device)
        # Sorting based on idx
        emb = embeddings.cpu().numpy()
        idx = indices.cpu()
        lab = labels.cpu().numpy()

        mask = np.argsort(idx)
        embeddings_sorted = emb[mask]
        idx_sorted = idx[mask]
        labels_sorted = lab[mask]

        # Appending results
        embeddings_list.append(embeddings_sorted)
        indices_list.append(idx_sorted)
        labels_list.append(labels_sorted)
        AE_list.append(AE)
        # Save embeddings, indices, and labels if flag is set.
        
        if save:
            np.save(os.path.join(save_dir, f'embeddings_trial_{i+1}_dim{latent_dim}.npy'), embeddings.cpu().numpy())
            np.save(os.path.join(save_dir, f'indices_trial_{i+1}_dim{latent_dim}.npy'), indices.cpu().numpy())
            np.save(os.path.join(save_dir, f'labels_trial_{i+1}_dim{latent_dim}.npy'), labels.cpu().numpy())

    return AE_list, embeddings_list, indices_list, labels_list, train_loss, test_loss

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


def train_decoder_on_relreps(anchors, train_loader, test_loader, model, num_epochs=5, lr=1e-3, device='cuda', verbose=False):
    """
    Trains the decoder using on-the-fly computed relative representations.
    For each batch, the latent codes are computed by passing x through the frozen encoder.
    Then, using the provided set of anchors and the batch’s latent codes,
    the cosine similarities are computed (i.e. the relative representation) which is passed to the decoder.

    Args:
        anchors (np.ndarray or torch.Tensor): Contains anchor embeddings. Expected shape is [A, latent_dim].
        train_loader (DataLoader): DataLoader for the train set.
        test_loader (DataLoader): DataLoader for the test set.
        model (Autoencoder): An instance of the autoencoder with a frozen encoder; only the decoder will be trained.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to perform training ('cpu' or 'cuda').
        verbose (bool): If True, prints loss details each epoch.

    Returns:
        model: The updated model.
        train_loss_list (list of float): Average training loss per epoch.
        test_loss_list (list of float): Average testing loss per epoch.
    """

    # Freeze encoder parameters.
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Convert anchors to a torch tensor if they are not already,
    # converting through a numpy array to avoid creating an extra dimension.
    if not torch.is_tensor(anchors):
        anchors = torch.tensor(np.array(anchors), dtype=torch.float32, device=device)
    else:
        anchors = anchors.to(device, dtype=torch.float32)
    
    # Ensure anchors is 2D. If it has extra dimensions, squeeze them out.
    while anchors.dim() > 2:
        anchors = anchors.squeeze(0)
    
    # Normalize the anchors along the latent dimension.
    anchors_norm = F.normalize(anchors, dim=1)
    num_anchors = anchors_norm.shape[0]
    
    # TODO: Should be able to delete this
    # Reinitialize the decoder so it accepts an input of size [num_anchors] and outputs a flattened image (28*28 = 784)
    model.decoder = torch.nn.Sequential(
        torch.nn.Linear(num_anchors, 128),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Linear(128, 784),
        torch.nn.Sigmoid()
    )
    
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.to(device)
    
    train_loss_list = []
    test_loss_list = []
    
    # Outer epoch loop with progress bar.
    epoch_iter = tqdm(range(num_epochs), desc="Decoder Training Epochs", unit="epoch")
    for epoch in epoch_iter:
        model.train()
        epoch_train_loss = 0.0
        
        # Training phase with progress bar (per batch)
        train_iter = tqdm(train_loader, desc="Training Batches", unit="batch", leave=False)
        for x, _ in train_iter:
            x = x.to(device)
            with torch.no_grad():
                latent = model.encoder(x)  # [batch_size, latent_dim]
            latent_norm = F.normalize(latent, dim=1)
            rel_batch = torch.mm(latent_norm, anchors_norm.t())  # [batch_size, num_anchors]
            
            #### TODO: WRITE TEST THAT COMPARES THESE RELREPS TO THE ACTUAL RELREPS
            
            output = model.decoder(rel_batch)
            x_flat = x.view(x.size(0), -1)
            loss = criterion(output, x_flat)
            epoch_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.3e}")
        
        # Evaluation phase with progress bar (per batch)
        model.eval()
        epoch_test_loss = 0.0
        test_iter = tqdm(test_loader, desc="Evaluation Batches", unit="batch", leave=False)
        with torch.no_grad():
            for x_test, _ in test_iter:
                x_test = x_test.to(device)
                latent_test = model.encoder(x_test)
                latent_test_norm = F.normalize(latent_test, dim=1)
                rel_batch_test = torch.mm(latent_test_norm, anchors_norm.t())
                output_test = model.decoder(rel_batch_test)
                x_test_flat = x_test.view(x_test.size(0), -1)
                loss_test = criterion(output_test, x_test_flat)
                epoch_test_loss += loss_test.item()
        avg_test_loss = epoch_test_loss / len(test_loader)
        test_loss_list.append(avg_test_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} - Test Loss: {avg_test_loss:.3e}")
        
        epoch_iter.set_postfix({"Train Loss": f"{avg_train_loss:.3e}", "Test Loss": f"{avg_test_loss:.3e}"})
    
    return model, train_loss_list, test_loss_list