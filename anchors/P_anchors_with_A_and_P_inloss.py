import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AnchorSelector(nn.Module):
    def __init__(self, N, N_anchors):
        super().__init__()
        # Use raw parameters for anchors (P)
        self.P = nn.Parameter(torch.randn(N_anchors, N))
    
    def forward(self, X):
        """
        X: [N, D] embeddings.
        Returns:
          anchors: [N_anchors, D] computed as weighted combinations of X.
        """
        # Use P directly (instead of softmax(Q))
        anchors = self.P @ X  # [N_anchors, D]
        return anchors

# Cosine similarity based diversity loss.
def diversity_loss(anchors, exponent=0.5):
    anchors_norm = F.normalize(anchors, p=2, dim=1)
    sim_matrix = anchors_norm @ anchors_norm.t()
    idx = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
    cosine_sim_values = abs(sim_matrix[idx[0], idx[1]])
    return torch.mean(cosine_sim_values ** exponent)

def coverage_loss(anchors, embeddings):
    anchors_norm = F.normalize(anchors, p=2, dim=1)
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim = abs(emb_norm @ anchors_norm.t())
    min_dists, _ = torch.min(sim, dim=1)
    return -torch.mean(min_dists)

def anchor_size_loss(anchors, target=1.0):
    # Penalize when anchors have norms below the target.
    current_sizes = torch.norm(anchors, dim=1)
    loss = torch.mean(torch.abs(target - current_sizes) ** 2)
    return loss

def parameter_magnitude_loss(anchor_selector):
    # Penalize for large magnitude in the parameter matrix P.
    return torch.norm(anchor_selector.P, p=2)

def optimize_anchors(anchor_selector, embeddings, epochs=100, lr=1e-3,
                     coverage_weight=1.0, diversity_weight=1.0,
                     anti_collapse_w=1.0, anchor_size_w=.0010, param_reg_w=.001,
                     exp=1, verbose=True):
    optimizer = torch.optim.Adam(anchor_selector.parameters(), lr=lr)
    for epoch in range(epochs):
        anchors = anchor_selector(embeddings)
        loss_cov = coverage_loss(anchors, embeddings)
        loss_div = diversity_loss(anchors, exponent=exp)
        loss_anchor_size = anchor_size_loss(anchors)
        loss_param = parameter_magnitude_loss(anchor_selector)
        loss = (diversity_weight * loss_div +
                coverage_weight * loss_cov +
                anchor_size_w * loss_anchor_size +
                param_reg_w * loss_param)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: total_loss={loss.item():.7f}, "
                  f"cov={loss_cov.item()*coverage_weight:.7f}, "
                  f"div={loss_div.item()*diversity_weight:.7f}, "
                  f"anchor_size={loss_anchor_size.item()*anchor_size_w:.7f}, "
                  f"param_reg={loss_param.item()*param_reg_w:.7f}")
    return anchor_selector(embeddings)

def get_optimized_anchors(emb, anchor_num, epochs=50, lr=1e-1,
                          coverage_weight=1.0, diversity_weight=1.0,
                          anti_collapse_w=1.0, anchor_size_w=1.0, param_reg_w=1.0,
                          exponent=1, verbose=True, device='cpu'):
    print("Optimizing P anchors...")
    X_first = emb[0]
    X_first_tensor = torch.from_numpy(X_first).to(device)
    anchor_selector = AnchorSelector(N=X_first_tensor.shape[0], N_anchors=anchor_num).to(device)
    optimize_anchors(anchor_selector, X_first_tensor, epochs=epochs, lr=lr,
                     coverage_weight=coverage_weight, diversity_weight=diversity_weight,
                     anti_collapse_w=anti_collapse_w, anchor_size_w=anchor_size_w, param_reg_w=param_reg_w,
                     exp=exponent, verbose=verbose)
    P_anchors_list = []
    for emb_run in emb:
        X_tensor_run = torch.from_numpy(emb_run).to(device)
        anchors_run = anchor_selector(X_tensor_run)
        P_anchors_list.append(anchors_run.cpu().detach().numpy())
    return anchor_selector, P_anchors_list