import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

# AutoEncoder Class
class Autoencoder(nn.Module):
    """
    Autoencoder with a bottleneck of size 2 that maps MNIST images to a 2D latent space.
    Includes training, evaluation, and embedding extraction methods.
    """
    
    ### NOTES ###
    # Might have to use batchnorm to impose a structure on the latent space

    def __init__(self, latent_dim=2, hidden_size=128):
        super().__init__()
        # Encoder layers
        # 784 -> 128 -> 2
        encoder_layers = [
            nn.Linear(28 * 28, hidden_size), # asuming size 28x28 of the images
            nn.Sigmoid(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim), # the size 2 bottleneck layer
            nn.BatchNorm1d(latent_dim)
        ]
        self.encoder = nn.Sequential(*encoder_layers) # '*' is unpacking the list into it's elements

        # Decoder layers
        # 2 -> 128 -> 784
        decoder_layers = [
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 28 * 28),
            nn.Sigmoid() # normalize outputs to [0, 1] - grayscale
        ]
        self.decoder = nn.Sequential(*decoder_layers)

        # Classifier Layers
        # 2 -> 128 -> 

    def encode(self, x):
        """
        Encodes an input batch (e.g., MNIST images) into the latent space.
        
        Args:
            x (Tensor): Input images of shape [batch_size, 784].
        Returns:
            z (Tensor): Encoded latent vectors of shape [batch_size, latent_dim].
        """
        return self.encoder(x)


    def decode(self, z):
        """
        Decodes latent vectors back to the original image space.
        
        Args:
            z (Tensor): Latent vectors of shape [batch_size, latent_dim].
        Returns:
            x_rec (Tensor): Reconstructed images of shape [batch_size, 784].
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Complete forward pass: encode then decode.
        
        Args:
            x (Tensor): Input images.
        Returns:
            reconstructed (Tensor): Reconstructed images of the same shape as x.
        """
        return self.decode(self.encode(x))

    def train_one_epoch(self, train_loader, optimizer, criterion, device='cuda'):
        """
        Performs one epoch of training.
        
        Args:
            train_loader (DataLoader): DataLoader for the training set.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
            criterion: Loss function (e.g., MSELoss, BCELoss).
            device (str): 'cpu' or 'cuda' device.
        
        Returns:
            epoch_loss (float): Average loss across this training epoch.
        """
        loss_total = 0.0
        self.train()

        for x, _ in train_loader:
            x = x.to(device)
            # Computing loss
            reconstructed = self.forward(x)
            loss = criterion(reconstructed, x)
            loss_total += loss.item()
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = loss_total / len(train_loader) # Computing average loss in epoch
        return epoch_loss

    def evaluate(self, data_loader, criterion, device='cuda'):
        """
        Evaluates the autoencoder on a given dataset (test or validation).
        
        Args:
            data_loader (DataLoader): DataLoader for the evaluation set.
            criterion: Loss function for reconstruction.
            device (str): 'cpu' or 'cuda'.
        
        Returns:
            eval_loss (float): Average reconstruction loss on this dataset.
        """
        self.eval() # Disable gradient computation
        loss_total = 0.0
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                # Computing loss
                reconstructed = self.forward(x)
                loss = criterion(reconstructed, x)
                loss_total += loss.item()

        eval_loss = loss_total / len(data_loader) # Computing average evaluation loss
        return eval_loss

    def fit(self, train_loader, test_loader, num_epochs, lr=1e-3, device='cuda', verbose=True):
        """
        High-level method to train the autoencoder for a given number of epochs.
        It orchestrates optimizer setup, training loop, and evaluation per epoch.
        
        Args:
            train_loader (DataLoader): DataLoader for training set.
            test_loader (DataLoader): DataLoader for test/validation set.
            num_epochs (int): Number of epochs.
            lr (float): Learning rate for the optimizer.
            device (str): 'cpu' or 'cuda'.
            verbose (bool): Each epoch prints loss if True
        
        Returns:
            train_losses (list of float): Loss for each training epoch.
            test_losses (list of float): Loss for each test epoch.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.MSELoss() # BCE loss may be better but gets error

        train_loss_list = []
        test_loss_list = []
        # Fitting the model
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            # Train loss
            train_loss = self.train_one_epoch(train_loader, optimizer, criterion=loss_function,device=device)
            train_loss_list.append(train_loss)
            # Test loss
            test_loss = self.evaluate(test_loader, criterion=loss_function, device=device)
            test_loss_list.append(test_loss)
            if verbose:
                print(f'Epoch #{epoch}')
                print(f'Train Loss = {train_loss:.3e} --- Test Loss = {test_loss:.3e}')
        
        return train_loss_list, test_loss_list

    def get_latent_embeddings(self, data_loader, device='cpu'):
        """
        Passes the entire dataset through the encoder to extract latent vectors.
        
        Args:
            data_loader (DataLoader): DataLoader for the dataset to encode.
            device (str): 'cpu' or 'cuda'.
        
        Returns:
            embeddings (Tensor): Concatenated latent vectors of shape [N, latent_dim].
            (indices, labels) (tuple of Tensors): Unique indices and corresponding labels for each sample.
        """
        embeddings = []
        indices = []
        labels = []

        self.eval()  # Disable gradient computation

        with torch.no_grad():
            for x, (idx, lab) in data_loader:
                x = x.to(device)
                z = self.encode(x)
                embeddings.append(z)
                indices.append(idx)
                labels.append(lab)
        
        embeddings_concat = torch.cat(embeddings, dim=0)
        indices_concat = torch.cat(indices, dim=0)
        labels_concat  = torch.cat(labels, dim=0)

        return embeddings_concat, indices_concat, labels_concat

import torch.nn as nn

class AEClassifier(nn.Module):
    """
    Classifier that reuses the encoder functionality (as in the Autoencoder)
    and adds a classification head for predicting labels.
    Inherits directly from nn.Module.
    """
    def __init__(self, latent_dim=2, hidden_size=128, num_classes=10):
        super().__init__()
        # Encoder layers (same as in the Autoencoder)
        encoder_layers = [
            nn.Linear(28 * 28, hidden_size),  # assuming 28x28 images
            nn.Sigmoid(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, latent_dim),  # bottleneck
            nn.BatchNorm1d(latent_dim)
        ]
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Classifier head layers
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def encode(self, x):
        # Flatten input if necessary (e.g., from images to vector)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return z

    def forward(self, x):
        z = self.encode(x)
        logits = self.classifier(z)
        return logits

    def train_one_epoch(self, train_loader: DataLoader, optimizer, criterion, device='cuda'):
        loss_total = 0.0
        self.train()
        for x, y_tuple in train_loader:
            x = x.to(device)
            # Unpack y tuple to get the label only.
            _, y = y_tuple  
            y = y.to(device)
            logits = self.forward(x)
            loss = criterion(logits, y)
            loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return loss_total / len(train_loader)

    def evaluate(self, data_loader: DataLoader, criterion, device='cuda'):
        self.eval()
        loss_total = 0.0
        with torch.no_grad():
            for x, y_tuple in data_loader:
                x = x.to(device)
                _, y = y_tuple
                y = y.to(device)
                logits = self.forward(x)
                loss = criterion(logits, y)
                loss_total += loss.item()
        return loss_total / len(data_loader)

    def accuracy(self, data_loader: DataLoader, device='cuda'):
        """
        Computes the accuracy (percentage of correct predictions) on a dataset.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y_tuple in data_loader:
                x = x.to(device)
                _, y = y_tuple
                y = y.to(device)
                logits = self.forward(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total if total > 0 else 0.0

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, num_epochs, lr=1e-3, device='cuda', verbose=True):
        """
        High-level method to train the classifier for a given number of epochs.
        Uses CrossEntropyLoss for classification.
        
        Returns:
            train_losses (list of float): Loss for each training epoch.
            test_losses (list of float): Loss for each test epoch.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()
        train_loss_list = []
        test_loss_list = []
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            train_loss = self.train_one_epoch(train_loader, optimizer, loss_function, device=device)
            train_loss_list.append(train_loss)
            test_loss = self.evaluate(test_loader, loss_function, device=device)
            test_loss_list.append(test_loss)
            if verbose:
                acc = self.accuracy(test_loader, device=device)
                print(f"Epoch {epoch}: Train Loss = {train_loss:.3e}, Test Loss = {test_loss:.3e}, Test Acc = {acc*100:.2f}%")
        return train_loss_list, test_loss_list
    
    def get_latent_embeddings(self, data_loader, device='cuda'):
        """
        Passes the entire dataset through the encoder to extract latent vectors.
        
        Args:
            data_loader (DataLoader): DataLoader for the dataset to encode.
            device (str): 'cpu' or 'cuda'.
        
        Returns:
            embeddings (Tensor): Concatenated latent vectors of shape [N, latent_dim].
            (indices, labels) (tuple of Tensors): Unique indices and corresponding labels for each sample.
        """
        embeddings = []
        indices = []
        labels = []

        self.eval()  # Disable gradient computation

        with torch.no_grad():
            for x, (idx, lab) in data_loader:
                x = x.to(device)
                z = self.encode(x)
                embeddings.append(z)
                indices.append(idx)
                labels.append(lab)
        
        embeddings_concat = torch.cat(embeddings, dim=0)
        indices_concat = torch.cat(indices, dim=0)
        labels_concat  = torch.cat(labels, dim=0)

        return embeddings_concat, indices_concat, labels_concat