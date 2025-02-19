import torch
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, base_model, head, anchors, distance_measure='cosine', head_type='decoder', device='cuda'):
        self.device = device
        self.rep_mode = distance_measure
        self.head_type = head_type

        self.base_model = base_model.to(device)
        for param in self.base_model.encoder.parameters():
            param.requires_grad = False

        self.head = head.to(device)
        self.anchors = self._prepare_anchors(anchors)
        
        if self.head_type == 'decoder':
            self.criterion = nn.MSELoss()
        elif self.head_type == 'classifier':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("head_type must be either 'decoder' or 'classifier'")

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
                latent = self.base_model.encoder(x)
            rel = self.compute_relreps(latent)
            output = self.head(rel)

            if self.head_type == 'decoder':
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
                latent = self.base_model.encoder(x)
                rel = self.compute_relreps(latent)
                output = self.head(rel)
                if self.head_type == 'decoder':
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

        for epoch in range(num_epochs):
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