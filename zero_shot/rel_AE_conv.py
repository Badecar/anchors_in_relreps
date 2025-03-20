import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

from models.build_encoder_decoder import build_dynamic_encoder_decoder

class rel_AE_conv_MNIST(nn.Module):
    """
    Zero-shot stitching decoder for MNIST.
    
    In zero-shot stitching we use a frozen, pretrained encoder to generate
    absolute embeddings which are then transformed into relative representations
    (via an anchor-based mechanism performed externally). This class uses those
    relative representations (of dimension `relative_output_dim`) as input and learns
    a decoder that reconstructs the original image, without re-training the encoder.
    
    Expected training data is a DataLoader yielding tuples: (relative_representation, target_image)
    where target_image is flattened (size 784) and relative_representation has shape [relative_output_dim].
    """
    def __init__(
        self,
        relative_output_dim: int,
        encoder_out_shape: torch.Size,
        n_channels: int,
        hidden_dims: list = (32, 64, 128, 256),
        latent_activation: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.relative_dim = relative_output_dim
        self.encoder_out_shape = encoder_out_shape  # Expected shape of conv features: [batch, C, H, W]
        self.n_channels = n_channels

        # Compute number of features (excluding batch dim)
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])

        # Build the mapping from relative representation to the convolutional feature space.
        # This acts like the "decoder_in" in AE_conv_MNIST.
        self.decoder_in = nn.Sequential(
            nn.Linear(relative_output_dim, encoder_out_numel),
            latent_activation() if latent_activation is not None else nn.Identity(),
        )

        # Build decoder using the same function as in AE_conv_MNIST.
        # We assume MNIST images (28x28, 1 channel). The returned tuple is (encoder, encoder_out_shape, decoder).
        _, _, self.decoder = build_dynamic_encoder_decoder(
            width=28,
            height=28,
            n_channels=n_channels,
            hidden_dims=hidden_dims,
            activation=latent_activation,
            remove_encoder_last_activation=False,
        )

    def loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """
        Computes the loss function in the same manner as AE_conv.
        """
        predictions = model_out  # In this AE, model_out is the reconstructed output.
        targets = batch["image"]
        mse = F.mse_loss(predictions, targets, reduction="mean")
        log_sigma_opt = 0.5 * mse.log()
        r_loss = log_sigma_opt
        r_loss = r_loss.sum()
        loss = r_loss
        return {
            "loss": loss,
            "reconstruction": r_loss.detach() / targets.shape[0],
        }

    def decode(self, relative_embedding): # For conv decoder
        """
        Decodes relative representations into flattened MNIST images.
        
        Args:
            relative_embedding (Tensor): shape [batch_size, relative_dim]
        Returns:
            x_rec (Tensor): flattened reconstructions with shape [batch_size, 784]
        """
        decoder_in = self.decoder_in(relative_embedding)
        # Reshape to conv feature map shape (omit batch dim from encoder_out_shape)
        decoder_in_conv = decoder_in.view(-1, *self.encoder_out_shape[1:])
        x_rec = self.decoder(decoder_in_conv)
        # Flatten the reconstructed image
        x_rec = x_rec.view(x_rec.size(0), -1)
        return x_rec
    
    def _decode(self, relative_embedding): # For fc decoder
        """
        Decodes relative representations into flattened MNIST images.
        
        Args:
            relative_embedding (Tensor): shape [batch_size, relative_dim]
        Returns:
            x_rec (Tensor): flattened reconstructions with shape [batch_size, 784]
        """
        latent = self.decoder_in(relative_embedding)
        x_rec = self.decoder(latent)
        return x_rec

    def forward(self, relative_embedding):
        """
        Forward pass: decode relative representations into flattened images.
        """
        return self.decode(relative_embedding)

    def train_one_epoch(self, train_loader, optimizer, device='cuda'):
        loss_total = 0.0
        self.train()
        for rel_emb, target in train_loader:
            rel_emb = rel_emb.to(device)
            target = target.to(device)
            reconstruction = self.forward(rel_emb)
            loss_dict = self.loss_function(reconstruction, {"image": target})
            loss = loss_dict["loss"]
            loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss = loss_total / len(train_loader)
        return epoch_loss

    def evaluate(self, data_loader, device='cuda'):
        self.eval()
        loss_total = 0.0
        with torch.no_grad():
            for rel_emb, target in data_loader:
                rel_emb = rel_emb.to(device)
                target = target.to(device)
                reconstruction = self.forward(rel_emb)
                loss_dict = self.loss_function(reconstruction, {"image": target})
                loss_total += loss_dict["loss"].item()
        eval_loss = loss_total / len(data_loader)
        return eval_loss

    def fit(self, train_loader, test_loader, num_epochs, lr=1e-3, device='cuda', verbose=True):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_loss_list, test_loss_list = [], []
        for epoch in tqdm(range(num_epochs), desc="Training Epochs", disable=not verbose):
            train_loss = self.train_one_epoch(train_loader, optimizer, device=device)
            train_loss_list.append(train_loss)
            test_loss = self.evaluate(test_loader, device=device)
            test_loss_list.append(test_loss)
            if verbose:
                print(f'Epoch #{epoch}')
                print(f'Train Loss = {train_loss:.3e} --- Test Loss = {test_loss:.3e}')
        return train_loss_list, test_loss_list