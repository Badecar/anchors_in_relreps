import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        # Assume the VAE decoder is defined by fc3 and fc4.
        # These layers map a latent vector (of dim=anchor_num) to the original image space.
        self.fc3 = vae.fc3  # already a Linear layer from latent_dim to hidden_size
        self.fc4 = vae.fc4  # Linear layer from hidden_size to input_dim
    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder that maps MNIST images into a latent space
    of given dimensions. The loss function is the sum of reconstruction loss
    (binary cross entropy) and a KL divergence loss weighted by beta.
    
    The KL loss encourages the latent code distribution to be close to N(0, I),
    which can help in separating classes.
    """
    def __init__(self,input_dim, latent_dim=2, hidden_size=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 784 -> hidden_size -> (mu, logvar)
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)
        
        # Decoder: latent_dim -> hidden_size -> 784
        self.fc3 = nn.Linear(latent_dim, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # Using sigmoid since MNIST images are normalized between 0 and 1.
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        # Reconstruction loss: we use binary cross entropy
        MSE = F.mse_loss(recon_x, x, reduction='mean')
        # KL-divergence term
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalize by batch size
        return (MSE + 0 * KLD)

    def train_one_epoch(self, train_loader: DataLoader, optimizer, beta=1.0, device='cuda'):
        self.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = self.forward(x)
            loss = self.loss_function(recon, x, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def evaluate(self, data_loader: DataLoader, beta=1.0, device='cuda'):
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                recon, mu, logvar = self.forward(x)
                loss = self.loss_function(recon, x, mu, logvar, beta=beta)
                total_loss += loss.item()
        return total_loss / len(data_loader)

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs, lr=1e-3, beta=1.0, device='cuda', verbose=True):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        test_losses = []
        for epoch in tqdm(range(num_epochs), desc="Training Epochs", disable=not verbose):
            train_loss = self.train_one_epoch(train_loader, optimizer, beta=beta, device=device)
            test_loss = self.evaluate(test_loader, beta=beta, device=device)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        return train_losses, test_losses

    def get_latent_embeddings(self, data_loader: DataLoader, device='cpu'):
        """
        Passes data through the encoder and returns the mean (mu) of the latent distribution.
        Sorting is performed based on the provided indices.
        """
        self.eval()
        embeddings = []
        indices = []
        labels = []
        with torch.no_grad():
            for x, (idx, lab) in data_loader:
                x = x.to(device)
                mu, _ = self.encode(x)
                embeddings.append(mu)
                indices.append(idx)
                labels.append(lab)
        embeddings = torch.cat(embeddings, dim=0)
        indices = torch.cat(indices, dim=0)
        labels = torch.cat(labels, dim=0)
        sorted_order = torch.argsort(indices)
        return (embeddings[sorted_order],
                indices[sorted_order],
                labels[sorted_order])
    
    def validate(self, data_loader, beta=1.0, device='cuda'):
        """
        Evaluates the VAE on the provided data_loader.
        
        Args:
            data_loader (DataLoader): DataLoader for the validation/test dataset.
            beta (float): Weight factor for the KL divergence loss.
            device (str): Device to run the evaluation on ('cpu' or 'cuda').
            
        Returns:
            tuple: (mean_loss, std_loss) computed over batches.
        """
        self.eval()
        losses = []
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                recon, mu, logvar = self(x)
                loss = self.loss_function(recon, x, mu, logvar, beta=beta)
                losses.append(loss.item())
        losses = np.array(losses)
        mean_loss = losses.mean()
        std_loss = losses.std()
        return mean_loss, std_loss
