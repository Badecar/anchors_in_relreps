import os
import numpy as np
from data import load_mnist_data
from utils import set_random_seeds

# Train AE
def train_AE(model, num_epochs=5, batch_size=256, lr=1e-3, device='cuda', latent_dim = 2, hidden_layer = 128, trials=1, use_test=True, save=False, verbose=False, train_loader=None, test_loader=None):
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
    embeddings_list = []
    indices_list = []
    labels_list = []
    AE_list = []
    acc_list = []

    # Create the directory to save embeddings if needed.
    if save:
        save_dir = os.path.join("models", "saved_embeddings", f"dim{latent_dim}")
        os.makedirs(save_dir, exist_ok=True)

    for i in range(trials):
        if verbose:
            print(f"Trial {i+1} of {trials}")
        # Create the data loaders
        # train_loader, test_loader = load_mnist_data(batch_size=batch_size)
        # Initialize and train the autoencoder
        AE = model(latent_dim=latent_dim, hidden_size=hidden_layer)
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
        if hasattr(model, "accuracy"):
            acc = AE.accuracy(test_loader, device)
            acc_list.append(acc)
            if verbose: print(f'Accuracy of the network on the test images: {acc:.2f}%')
        
        if save:
            np.save(os.path.join(save_dir, f'embeddings_trial_{i+1}_dim{latent_dim}.npy'), embeddings.cpu().numpy())
            np.save(os.path.join(save_dir, f'indices_trial_{i+1}_dim{latent_dim}.npy'), indices.cpu().numpy())
            np.save(os.path.join(save_dir, f'labels_trial_{i+1}_dim{latent_dim}.npy'), labels.cpu().numpy())

    return AE_list, embeddings_list, indices_list, labels_list, train_loss, test_loss, acc_list

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


def train_on_relrep(model, relreps, train_loader, test_loader, num_epochs=5, lr=1e-3, device='cuda', latent_dim = 2, verbose=False):
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr) # change model to anchors parameters

    loss_function = nn.MSELoss()
    model.train()

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)

            # Forward pass
            


            # Test loss
            test_loss = self.evaluate(test_loader, criterion=loss_function, device=device)
            test_loss_list.append(test_loss)
            if verbose:
                print(f'Epoch #{epoch}')
                print(f'Train Loss = {train_loss:.3e} --- Test Loss = {test_loss:.3e}')
