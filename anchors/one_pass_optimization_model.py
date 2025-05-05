import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

from models import build_dynamic_encoder_decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class one_pass_optimization(nn.Module):
    """Learnable-Anchor Relative Representation Decoder

    This module implements the pipeline discussed:
    1.  **Frozen embedding bank** `X` (shape ``[n, d]``) coming from a pre‑trained encoder.
    2.  **Learnable weight matrix** ``P  /in R^{m×n}`` whose rows (optionally
        soft‑maxed) linearly combine the embeddings into ``m`` anchors.
    3.  **Relative‑representation layer** that compares a *batch* of embeddings
        against the anchors via either cosine similarity *or* a Mahalanobis metric.
    4.  **Decoder** that maps the relative representation back to image space.

    Gradients flow through the *decoder* and *P*; the embed bank is a frozen buffer.
    """

    def __init__(
            self,
            embed_banks: Union[torch.Tensor, np.ndarray, Sequence[Union[torch.Tensor, np.ndarray]]],
            num_anchors: int,
            device: torch.device,
            similarity: str = "mahalanobis",
            hidden_dims=(32, 64),
            row_softmax: bool = True,
            image_shape: tuple[int, int, int] = (1, 28, 28),
        ):
        super().__init__()
        assert similarity in {"cosine", "mahalanobis"}

        # ------------------------------------------------------------
        # 1. Normalise input → list[Tensor] on the same device
        # ------------------------------------------------------------
        if isinstance(embed_banks, (torch.Tensor, np.ndarray)):
            embed_banks = [embed_banks]                   # wrap single bank
        bank_list: list[torch.Tensor] = []
        for bank in embed_banks:
            if isinstance(bank, np.ndarray):
                bank = torch.from_numpy(bank)
            bank_list.append(bank.float().to(device))     # [N_i, d]

        # basic checks
        d = bank_list[0].shape[1]
        assert all(b.shape[1] == d for b in bank_list), "all banks must share same embedding dim"

        self.embed_banks = bank_list         # keep as plain Python list
        self.num_banks   = len(bank_list)
        self.d           = d
        self.m           = num_anchors
        self.row_softmax = row_softmax
        self.similarity  = similarity

        # ------------------------------------------------------------
        # 2. Pre-compute Σ⁻¹ for every bank  →  stack into one buffer
        # ------------------------------------------------------------
        inv_covs = []
        eps = 1e-6
        for B in bank_list:                  # B: [N, d]
            X   = B - B.mean(0, keepdim=True)
            cov = (X.T @ X) / max(1, len(B) - 1)               # [d,d]
            # eigendecomp for stability: cov = V diag(λ) Vᵀ
            λ, V = torch.linalg.eigh(cov)                       # ascending λ
            inv_cov = V @ torch.diag(1.0 / (λ + eps)) @ V.T     # Σ⁻¹
            inv_covs.append(inv_cov)

        self.register_buffer("inv_cov_stack",
                             torch.stack(inv_covs, dim=0))      # [R, d, d]

        # ------------------------------------------------------------
        # 3. Learnable P (shared across banks)
        # ------------------------------------------------------------
        self.raw_P = nn.Parameter(torch.randn(self.m, bank_list[0].shape[0]) * 1e-2)
        # ------------------------------------------------------------------
        # Default decoder (if one isn’t supplied)
        # ------------------------------------------------------------------
    
        # Build conv decoder mirroring AE_conv
        _, conv_shape, conv_dec = build_dynamic_encoder_decoder(
            width=image_shape[2], height=image_shape[1],
            n_channels=image_shape[0],
            hidden_dims=hidden_dims,
            activation=nn.GELU,
            remove_encoder_last_activation=False
        )
        conv_numel = math.prod(conv_shape[1:])        # C*H*W
        class _RelRepConvDecoder(nn.Module):
            def __init__(self, m, conv_numel, conv_shape, conv_dec):
                super().__init__()
                self.fc = nn.Sequential(nn.Linear(m, conv_numel), nn.GELU())
                self.conv_shape = conv_shape[1:]
                self.conv_dec  = conv_dec
            def forward(self, r):
                x = self.fc(r)
                x = x.view(x.size(0), *self.conv_shape)
                x = self.conv_dec(x)
                return x.view(x.size(0), -1)          # flatten to 784
        decoder = _RelRepConvDecoder(self.m, conv_numel, conv_shape, conv_dec)
        self.decoder = decoder.to(device)
        self._out_shape = None  # decoder decides its own output shape


    def _select_bank(self, bank_idx: int):
        return self.embed_banks[bank_idx], self.inv_cov_stack[bank_idx]
    
    # ----------------------------------------------------------------------
    # Helper: get the *current* P (optionally row‑soft‑maxed)
    # ----------------------------------------------------------------------
    def _get_P(self):
        if self.row_softmax:
            return torch.softmax(self.raw_P, dim=1)  # each row sums to 1
        return self.raw_P

    # ----------------------------------------------------------------------
    # Anchors A = P @ X
    # ----------------------------------------------------------------------
    def get_anchors(self, bank_idx: int = 0):
        X, _ = self._select_bank(bank_idx)
        P = torch.softmax(self.raw_P, dim=1) if self.row_softmax else self.raw_P
        return P @ X                         # [m, d]
    # ----------------------------------------------------------------------
    # Relative‑rep computations
    # ----------------------------------------------------------------------
    @staticmethod
    def _relrep_cosine(e: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # e: [b, d]  A: [m, d]
        e_n = F.normalize(e, dim=1)
        A_n = F.normalize(A, dim=1)
        return e_n @ A_n.T             # [b, m]

    def _relrep_mahalanobis(self, e: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Negative Mahalanobis distance as similarity
        diff = e.unsqueeze(1) - A.unsqueeze(0)          # [b, m, d]
        M = torch.einsum("bmd,dd,bmd->bm", diff, self.inv_cov, diff)
        return -M                                       # similarity (higher is better)

    def relrep(self, e: torch.Tensor, bank_idx: int = 0) -> torch.Tensor:
        A, inv_cov = self.get_anchors(bank_idx), self.inv_cov_stack[bank_idx]
        if self.similarity == "cosine":
            e_n, A_n = F.normalize(e, dim=1), F.normalize(A, dim=1)
            return e_n @ A_n.T                                   # cosine
        # Mahalanobis
        diff = e.unsqueeze(1) - A.unsqueeze(0)                  # [B,m,d]
        return -torch.einsum("bmd,dd,bmd->bm", diff, inv_cov, diff)

    # ----------------------------------------------------------------------
    # Forward pass: embeddings → reconstruction
    # ----------------------------------------------------------------------
    def forward(self, embeddings: torch.Tensor, bank_idx: int = 0):
        r     = self.relrep(embeddings, bank_idx)               # [B,m]
        recon = self.decoder(r)
        return recon        # ------------------------------------------------------------------
    # Utility: visualise reconstruction vs. original
    # ------------------------------------------------------------------
    def visualize_reconstruction(
        self,
        embedding: torch.Tensor,
        image: torch.Tensor,
        cmap: str = "gray",
        title: str = "Reconstruction vs. Original",
    ) -> None:
        """Plot reconstructed image next to original.

        Args:
            embedding: Single embedding vector (shape [d] or [1,d]). Must be on same device.
            image: Corresponding flattened image tensor (shape [784] or [1,784]).
            cmap: Colormap for imshow.
            title: Figure title.
        """
        self.eval()
        with torch.no_grad():
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            if image.dim() == 1:
                image = image.unsqueeze(0)
            device = next(self.parameters()).device
            embedding = embedding.to(device)
            image = image.to(device)

            recon = self.forward(embedding)

            real_img  = image.view(-1, 28, 28).cpu()
            recon_img = recon.view(-1, 28, 28).cpu()

        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        axes[0].imshow(real_img[0], cmap=cmap)
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(recon_img[0], cmap=cmap)
        axes[1].set_title("Reconstruction")
        axes[1].axis("off")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    # ------------------------------------------------------------------
    # Save model parameters
    # ------------------------------------------------------------------
    def save_model(self, path: str) -> None:
        """Save the model's state_dict (including buffers) to *path*."""
        torch.save(self.state_dict(), path)
    def load_model(self, path: str, map_location: Optional[torch.device] = None) -> None:
        """Load a saved state_dict from *path* into this model instance.

        Args:
            path: File path to a ``.pth`` / ``.pt`` checkpoint saved via :py:meth:`save_model`.
            map_location: Optional torch device mapping (defaults to the model's device).
        """
        device = map_location or next(self.parameters()).device
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
def compute_whitener(embeddings: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Return Σ^{-½}  (d×d) so that  x_white = x @ Σ^{-½}.
    embeddings : [N, d]  on same device as model
    """
    # (1) sample covariance Σ = Xᵀ X / (N-1)
    X   = embeddings - embeddings.mean(0, keepdim=True)     # center
    cov = (X.T @ X) / (X.size(0) - 1)                       # [d,d]

    # (2) eigen-decomposition  cov = V diag(λ) Vᵀ
    eigvals, eigvecs = torch.linalg.eigh(cov)               # λ ascending

    # (3) Σ^{-½} = V diag(λ^{-½}) Vᵀ
    inv_sqrt = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + eps)) @ eigvecs.T
    return inv_sqrt    
def compute_covariance_matrix(embeddings):
    """
    Computes the sample covariance matrix for embeddings.
    
    embeddings: Tensor of shape [N, D]
    
    Returns: Tensor of shape [D, D]
    """
    mean = embeddings.mean(dim=0, keepdim=True)
    centered = embeddings - mean
    cov = (centered.t() @ centered) / (embeddings.size(0) - 1)
    return cov
def coverage_loss(anchors_w, embeds_w):
    # Euclidean distances after whitening
    dists = torch.cdist(embeds_w, anchors_w)        # [N, m]
    return dists.min(dim=1).values.mean()

def diversity_loss(anchors_w):
    # pairwise distances between anchors only
    d = torch.pdist(anchors_w, p=2)                 # [m*(m-1)/2]
    return -(d).mean()  
# ---------- build one whitener (Σ−½) from a bank -------------------
def make_whitener(bank: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    X   = bank - bank.mean(0, keepdim=True)
    cov = (X.T @ X) / max(1, len(bank) - 1)           # [d,d]
    λ, V = torch.linalg.eigh(cov)
    return V @ torch.diag(1.0 / torch.sqrt(λ + eps)) @ V.T   # Σ−½

# ---------- coverage & diversity on whitened tensors ---------------
def coverage_loss_wh(anchors_w: torch.Tensor, embeds_w: torch.Tensor):
    # mean(min distance)  –> we *negate* so lower dist = higher loss
    return torch.cdist(embeds_w, anchors_w).min(1).values.mean()

def diversity_loss_wh(anchors_w: torch.Tensor, exponent: float = 1.0):
    # mean pairwise anchor distance  –> negate so small ⇢ high loss
    return -torch.pdist(anchors_w, p=2).pow(exponent).mean()

def pretrain_P(
    model: nn.Module,
    epochs: int = 500,
    lr: float = 1e-2,
    device: str | torch.device = "cuda",
    λ_cov: float = 0.95,
    λ_div: float = 0.05,
):

    device   = torch.device(device)
    model    = model.to(device)
    embeds   = model.embed_banks          # [N, d]
    embeds   = embeds[0]
    # freeze everything except P
    for p in model.parameters():  p.requires_grad_(False)
    model.raw_P.requires_grad_(True)

    # build whitening matrix once
    L = compute_whitener(embeds).to(device)   # d×d
    embeds_w  = embeds  @ L.T                 # (N,d)
    opt = torch.optim.Adam([model.raw_P], lr=lr)

    for _ in tqdm(range(epochs), desc="pretraining P"):
        anchors   = model.get_anchors()             # [m, d]
        anchors_w = anchors @ L.T 
        anchors_w = anchors @ L.T                   # whiten anchors

        loss = (λ_cov * coverage_loss(anchors_w, embeds_w) +
                λ_div * diversity_loss(anchors_w))

        opt.zero_grad()
        loss.backward()
        opt.step()

    for p in model.parameters():  p.requires_grad_(True)

def relrep_alignment_loss(
    embeddings_list,
    model, 
    bank_idx,
) -> torch.Tensor:
    """
    Compute MSE loss between two relative-representation spaces.
    
    Args:
        embeddings1: [B, d] from space 1
        embeddings2: [B, d] from space 2
        anchors1:    [m, d] anchors for space 1
        anchors2:    [m, d] anchors for space 2
    Returns:
        scalar MSE( R1, R2 )
    """
    R1 = model.relrep(embeddings_list[0], bank_idx[0])  # [B, m]
    R2 = model.relrep(embeddings_list[0], bank_idx[0])  # [B, m]
    return F.mse_loss(R1, R2)


def train_relrep_decoder(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    id_lookup,                 # list of lookup tensors  id→row
    banks: list[torch.Tensor], # same order as id_lookup
    whiteners: list[torch.Tensor],
    w_cd: float = 0.1,
    w_alignment : float = 0.3, 
    epochs: int = 10,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
    verbose : bool = False
):
    model = model.to(device).train()
    embed_stack = torch.stack(banks, 0).to(device)      # [R,N,d]  on device
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(1, epochs + 1), disable=not verbose):
        total, count = 0.0, 0
        for imgs, (ids, _) in dataloader:
            imgs  = imgs.to(device)
            # pick a random embedding space this batch
            bank_idx = random.sample(range(len(banks)), 2)

            rows1  = id_lookup[bank_idx[0]][ids.to(device)].to(device)      # [B]
            emb1   = embed_stack[bank_idx[0], rows1].to(device)              # [B,d]
            rows2  = id_lookup[bank_idx[1]][ids.to(device)].to(device)      # [B]
            emb2   = embed_stack[bank_idx[1], rows1].to(device)              # [B,d]
            
            # forward with matching bank
            recon = model(embeddings=emb1, bank_idx=bank_idx[0])
            mse   = F.mse_loss(recon, imgs)

            # alignment_loss
            alignment_loss = relrep_alignment_loss(embeddings_list=[emb1, emb2], model=model, bank_idx=bank_idx)

            # coverage + diversity on whitened anchors
            L         = whiteners[bank_idx[0]]                  # [d,d]
            anchors_w = model.get_anchors(bank_idx[0]) @ L.T    # [m,d]
            embeds_w  = emb1 @ L.T                       # [B,d]

            cov_loss  = coverage_loss_wh(anchors_w, embeds_w)
            div_loss  = diversity_loss_wh(anchors_w)

            loss = mse + w_cd * (9/10* cov_loss + 1/10 * div_loss) + w_alignment*alignment_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            total  += loss.item() * imgs.size(0)
            count  += imgs.size(0)

        print(f"[{epoch}/{epochs}]  avg loss = {total/count:.6f}")