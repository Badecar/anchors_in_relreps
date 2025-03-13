import os
import numpy as np
import torch
from .autoencoder import Autoencoder, AEClassifier
from .VAE import VariationalAutoencoder




def get_save_dir(model, latent_dim):
    if model == AEClassifier:
            m = 'AEClassifier'
    elif model == Autoencoder:
            m = 'AE'
    elif model == VariationalAutoencoder:
            m = 'VAE'
    save_dir_emb = os.path.join("experiments", "saved_embeddings", m, f"dim{latent_dim}")
    save_dir_AE = os.path.join("experiments", "saved_models", m, f"dim{latent_dim}")
    os.makedirs(save_dir_AE, exist_ok=True)
    os.makedirs(save_dir_emb, exist_ok=True)
    return save_dir_emb, save_dir_AE

def load_saved_emb(model, trials=1, latent_dim=int, save_dir=None):
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
    print("Loading saved embeddings and models")
    if save_dir is None:
        save_dir, _ = get_save_dir(model, latent_dim)
    
    embeddings_list = []
    indices_list = []
    labels_list = []
    
    for i in range(trials):
        emb_path = os.path.join(save_dir, f'embeddings_trial_{i+1}_dim{latent_dim}.npy')
        idx_path = os.path.join(save_dir, f'indices_trial_{i+1}_dim{latent_dim}.npy')
        lab_path = os.path.join(save_dir, f'labels_trial_{i+1}_dim{latent_dim}.npy')
        
        # Check if the saved files exist, if not, throw an error and stop program execution.
        if not (os.path.exists(emb_path) and os.path.exists(idx_path) and os.path.exists(lab_path)):
            raise FileNotFoundError(f"Saved data not found for trial {i+1}. Expected files are missing in {save_dir}, file: {emb_path}.")
        else:
            # Load the saved .npy files
            embeddings = np.load(emb_path)
            indices = np.load(idx_path)
            labels = np.load(lab_path)

            embeddings_list.append(embeddings)
            indices_list.append(indices)
            labels_list.append(labels)
        
    return embeddings_list, indices_list, labels_list

def load_AE_models(model, trials=1, latent_dim=2, input_dim=28*28, hidden_layer=128, save_dir_AE=None, device='cuda'):
    """
    Loads a specified number of saved AE (or AEClassifier) models into a list.
    
    Args:
        model (class): The model class to load (e.g., Autoencoder or AEClassifier).
        trials (int): The number of saved model trials to load.
        latent_dim (int): The latent dimensionality (must match the saved models).
        hidden_layer (int): Size of the hidden layer.
        save_dir_AE (str): Directory where the saved models are located. 
                           If None, defaults to "models/saved_models/dim{latent_dim}".
        device (str): Device to load the model ('cpu' or 'cuda').
        
    Returns:
        list: A list of loaded models. Models not found will be skipped.
    """
    if save_dir_AE is None:
        _, save_dir_AE = get_save_dir(model, latent_dim)
    AE_list = []
    for i in range(1, trials + 1):
        model_path = os.path.join(save_dir_AE, f'ae_trial_{i}_dim{latent_dim}.pth')
        if os.path.exists(model_path):
            # For VAE, initialize with input_dim, hidden_dims and beta; otherwise, use hidden_size.
            if model.__name__ == "VariationalAutoencoder":
                # Ensure input_dim is provided; you might consider adding input_dim as a parameter to load_AE_models.
                # loaded_model = model(input_dim=input_dim, latent_dim=latent_dim,
                                    #  hidden_dims=[hidden_layer, hidden_layer // 2], beta=1)
                loaded_model = model(input_dim=input_dim, latent_dim=latent_dim,
                                     hidden_size=hidden_layer)                    
            else:
                loaded_model = model(latent_dim=latent_dim, hidden_size=hidden_layer, input_dim=input_dim)
            loaded_model.load_state_dict(torch.load(model_path, map_location=device))
            loaded_model.to(device)
            AE_list.append(loaded_model)
        else:
            print(f"Saved model for trial {i} not found at {model_path}. Skipping.")
    return AE_list