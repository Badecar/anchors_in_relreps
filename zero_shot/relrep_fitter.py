import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models import VAEDecoderWrapper, VariationalAutoencoder


def compute_relreps(latent, anchors, mode='cosine'):
    if mode == 'cosine':
        latent_norm = F.normalize(latent, dim=1)
        anchors_norm = F.normalize(anchors, dim=1)
        return torch.mm(latent_norm, anchors_norm.t())
    elif mode == 'euclidean':
        return torch.cdist(latent, anchors, p=2)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

class RelRepTrainer:
    def __init__(self, base_model, head, anchors, distance_measure='cosine', head_type='reconstructor', device='cuda'):
        self.device = device
        self.rep_mode = distance_measure
        self.head_type = head_type

        self.base_model = base_model.to(device)

        # Freeze the encoder:
        # If the model has an 'encoder' attribute, use it; otherwise freeze layers used in encoding.
        if hasattr(self.base_model, "encoder"):
            for param in self.base_model.encoder.parameters():
                param.requires_grad = False
        else:
            # Assuming a VAE structure with encode method using fc1, fc_mu, fc_logvar.
            for name, param in self.base_model.named_parameters():
                if any(layer in name for layer in ["fc1", "fc_mu", "fc_logvar"]):
                    param.requires_grad = False

        self.head = head.to(device)
        self.anchors = self._prepare_anchors(anchors)
        
        if self.head_type == 'reconstructor':
            self.criterion = nn.MSELoss()
        elif self.head_type == 'classifier':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("head_type must be either 'reconstructor' or 'classifier'")

    def _prepare_anchors(self, anchors):
        # Convert list of numpy arrays to one array before making a tensor.
        if not isinstance(anchors, np.ndarray):
            anchors = np.array(anchors)
        anchors = torch.tensor(anchors, dtype=torch.float32, device=self.device)
        while anchors.dim() > 2:
            anchors = anchors.squeeze(0)
        if self.rep_mode == 'cosine':
            anchors = F.normalize(anchors, dim=1)
        return anchors

    def compute_relreps(self, latent):
        return compute_relreps(latent, self.anchors, mode=self.rep_mode)

    def train_epoch(self, data_loader, optimizer):
        self.base_model.eval()  # Keep encoder frozen.
        self.head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in tqdm(data_loader, desc="Training Batches", leave=False):
            x = x.to(self.device)
            # If y is a tuple/list (like (index, label)), take the label at index 1.
            if isinstance(y, (tuple, list)):
                y = y[1]
            y = y.to(self.device)
            with torch.no_grad():
                if isinstance(self.base_model, VariationalAutoencoder):
                    # Use the full forward pass to sample stochastically
                    _, latent, _ = self.base_model(x)
                else:
                    latent = self.base_model.encoder(x)
            rel = self.compute_relreps(latent)
            output = self.head(rel)

            if self.head_type == 'reconstructor':
                x_flat = x.view(x.size(0), -1)
                loss = self.criterion(output, x_flat)
            else:  # classifier head
                loss = self.criterion(output, y)
                _, predicted = torch.max(output, 1)
                total_correct += (predicted == y).sum().item()
                total_samples += y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        if self.head_type == 'classifier':
            accuracy = total_correct / total_samples
            return avg_loss, accuracy
        else:
            return avg_loss

    def eval_epoch(self, data_loader):
        self.base_model.eval()
        self.head.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in tqdm(data_loader, desc="Evaluation Batches", leave=False):
                x = x.to(self.device)
                if isinstance(y, (tuple, list)):
                    y = y[1]
                y = y.to(self.device)
                if isinstance(self.base_model, VariationalAutoencoder):
                    # Use the full forward pass to sample stochastically
                    _, latent, _ = self.base_model(x)
                else:
                    latent = self.base_model.encoder(x)
                rel = self.compute_relreps(latent)
                output = self.head(rel)
                if self.head_type == 'reconstructor':
                    x_flat = x.view(x.size(0), -1)
                    loss = self.criterion(output, x_flat)
                else:
                    loss = self.criterion(output, y)
                    _, predicted = torch.max(output, 1)
                    total_correct += (predicted == y).sum().item()
                    total_samples += y.size(0)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        if self.head_type == 'classifier':
            accuracy = total_correct / total_samples
            return avg_loss, accuracy
        else:
            return avg_loss

    def fit(self, train_loader, test_loader, num_epochs, lr, verbose=True):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=lr)
        train_losses = []
        test_losses = []
        accuracies = []  # For classifier head

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            train_out = self.train_epoch(train_loader, optimizer)
            eval_out = self.eval_epoch(test_loader)
            
            if self.head_type == 'classifier':
                train_loss, train_acc = train_out
                test_loss, test_acc = eval_out
                accuracies.append(test_acc)
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}, Test Acc: {test_acc*100:.2f}%")
            else:
                train_loss = train_out
                test_loss = eval_out
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}")
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        
        if self.head_type == 'classifier':
            return train_losses, test_losses, accuracies
        else:
            return train_losses, test_losses, None
    
def validate_relhead(decoder, relrep, ground_truth, device='cuda', show=False):
    """
    Validates a relative decoder using precomputed latent representations.
    It decodes the latent embeddings (relative coordinates) with the given decoder and calculates
    the per-sample MSE (and its standard deviation) between the decoder's output and ground truth.
    Optionally, if show is True, displays 4 random original images and their reconstructions.
    
    Args:
        decoder (nn.Module): The trained relative decoder (e.g., the head returned from train_rel_head).
        latent (Tensor): Precomputed latent representations (relative coordinates) of shape [N, num_anchors].
        ground_truth (Tensor): Ground truth images. If not flattened, they will be flattened.
        device (str): Device to run on ('cpu' or 'cuda').
        show (bool): If True, display 4 random original images and their reconstructions.
        
    Returns:
        tuple: (mse_mean, mse_std) where mse_mean is the mean MSE, and mse_std its standard deviation.
    """

    decoder.eval()
    with torch.no_grad():
        relrep = relrep.to(device)
        output = decoder(relrep)
        # Flatten ground_truth if needed (e.g. images may be [N, C, H, W])
        if ground_truth.dim() > 2:
            ground_truth = ground_truth.view(ground_truth.size(0), -1)
        ground_truth = ground_truth.to(device)
        # Compute per-sample MSE
        criterion = nn.MSELoss(reduction='none')
        loss_per_sample = criterion(output, ground_truth).mean(dim=1)
        mse_mean = loss_per_sample.mean().item()
        mse_std = loss_per_sample.std().item()

        if show:
            # Choose 4 random indices from the available samples.
            num_samples = ground_truth.size(0)
            sample_indices = np.random.choice(num_samples, 4, replace=False)

            fig, axes = plt.subplots(4, 2, figsize=(8, 16))
            for i, idx in enumerate(sample_indices):
                # Reshape flattened images to 28x28 (assuming MNIST)
                orig = ground_truth[idx].view(28, 28).cpu().numpy()
                rec = output[idx].view(28, 28).cpu().numpy()
                axes[i, 0].imshow(orig, cmap="gray")
                axes[i, 0].set_title("Original")
                axes[i, 0].axis("off")
                axes[i, 1].imshow(rec, cmap="gray")
                axes[i, 1].set_title("Reconstruction")
                axes[i, 1].axis("off")
            plt.tight_layout()
            plt.show()

    return mse_mean, mse_std

## THIS USING THE MEAN OF THE VAE AND THE DECODER IS CLOSER TO AN AE DECODER ##
def train_rel_head(anchor_num, anchors, num_epochs, AE, train_loader, test_loader, device, acc, train_loss_AE, test_loss_AE, head_type='reconstructor', distance_measure='cosine', lr=1e-3, verbose=True, show_AE_loss=False):
    if head_type == 'reconstructor':
        if isinstance(AE, VariationalAutoencoder):
            head = VAEDecoderWrapper(AE)
        else:
            head = nn.Sequential(
                nn.Linear(anchor_num, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 28 * 28),
                nn.Sigmoid()
            )
    else:
        head = nn.Sequential(
            nn.Linear(anchor_num, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10)
        )

    # Instantiate the trainer with the chosen head.
    trainer = RelRepTrainer(
        base_model=AE,
        head=head,
        anchors=anchors,
        distance_measure=distance_measure,
        head_type=head_type,
        device=device
    )

    # Train the head using trainer.fit, etc.
    fit_results = trainer.fit(
        train_loader,
        test_loader,
        num_epochs,
        lr=lr,
        verbose=verbose)

    train_loss_relrepfit, test_loss_relrepfit, test_accuracies = fit_results

    if show_AE_loss:
        print("\nFull Autoencoder Training Losses per Epoch:")
        print("Train Losses:", train_loss_AE)
        print("Test Losses:", test_loss_AE)
        if trainer.head_type == 'classifier':
            print("AE Accuracy:", acc)

    print("\nDecoder-Only Training Losses per Epoch (using relative representations):")
    print("Train Losses:", train_loss_relrepfit)
    print("Test Losses:", test_loss_relrepfit)
    if trainer.head_type == 'classifier':
        print("Test Accuracy:", test_accuracies[-1])
    
    if trainer.head_type == 'reconstructor':
        print("\nShowing 4 random examples of original images and their reconstruction:")
        batch = next(iter(test_loader))
        images, _ = batch
        images = images.to(device)
        with torch.no_grad():
            if isinstance(AE, VariationalAutoencoder):
                _, latent, _ = AE(images)
            else:
                latent = AE.encoder(images)
            rel = trainer.compute_relreps(latent)
        validate_relhead(trainer.head, rel, images, device=device, show=True)
    
    return trainer.head