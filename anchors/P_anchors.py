import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AnchorSelector(nn.Module):
    def __init__(self, N, N_anchors):
        super().__init__()
        # Raw parameters for anchors (logits)
        self.Q = nn.Parameter(torch.randn(N_anchors, N))
    
    def forward(self, X):
        """
        X: [N, D] embeddings.
        Returns:
          anchors: [N_anchors, D] computed as weighted combinations of X.
        """
        # Enforce each row to sum to 1 (soft assignment)
        # You can control the temperature in softmax for sharper assignment.
        T = 0.05
        P = F.softmax(self.Q / T, dim=1)
        anchors = P @ X  # [N_anchors, D]
        return anchors, P

# NOTE: Eucl
# def diversity_loss(anchors, exponent=0.5, scale=1.0/np.sqrt(2)):
#     # anchors: [N_anchors, D]
#     # Compute pairwise distances (for example, cosine or Euclidean)
#     pdist_vals = torch.pdist(anchors, p=2)  # example with Euclidean
#     scaled_pdist_vals = pdist_vals * scale # scale to get on the same order of magnitude as cosine similarity
#     return -torch.mean(scaled_pdist_vals ** exponent)  # negative so maximizing diversity

# def coverage_loss(anchors, embeddings, scale=1.0/np.sqrt(2)):
#     # embeddings: [N, D] and anchors: [N_anchors, D]
#     # For each embedding, compute its distance to each anchor and take the minimum.
#     # Then, average over all embeddings.
#     # (Here we use squared Euclidean distance)
#     dists = torch.cdist(embeddings, anchors, p=2)  # [N, N_anchors]
#     scaled_dist = dists * scale 
#     min_dists, _ = torch.min(dists, dim=1)
#     return torch.mean(min_dists)

#NOTE: Cosine similarity version of losses
def diversity_loss(anchors, exponent=0.5):
    # Compute pairwise distances (for example, cosine or Euclidean)
    anchors_norm = F.normalize(anchors, p=2, dim=1)
    sim_matrix = anchors_norm @ anchors_norm.t()
    idx = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
    cosine_sim_values = abs(sim_matrix[idx[0], idx[1]])
    return -torch.mean(cosine_sim_values ** exponent)

def coverage_loss(anchors, embeddings):
    # For each embedding, compute its distance to each anchor and take the minimum.
    anchors_norm = F.normalize(anchors, p=2, dim=1)
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim = abs(emb_norm @ anchors_norm.t())
    min_dists, _ = torch.min(sim, dim=1)
    return torch.mean(min_dists)

def anti_collapse_loss(anchors):
    # return torch.mean(torch.abs(1 - torch.norm(anchors, dim=1)))
    return torch.mean(torch.relu(1 - torch.norm(anchors, dim=1)))


def optimize_anchors(anchor_selector, embeddings, epochs=100, lr=1e-3, coverage_weight=1.0, diversity_weight=1.0, anti_collapse_w=1.0, exp=1, verbose=True):
    """
    Optimize the Q parameters in AnchorSelector so that anchors = softmax(Q) @ embeddings
    minimize a combined loss of coverage and diversity.
    
    Args:
      anchor_selector (AnchorSelector): Instance to optimize.
      embeddings (Tensor): [N, D] embeddings.
      epochs (int): Number of optimization iterations.
      lr (float): Learning rate.
      coverage_weight (float): Weight for the coverage term.
      diversity_weight (float): Weight for the diversity term.
      verbose (bool): If set, print the loss every few epochs.
      
    Returns:
      anchors (Tensor): The optimized anchors matrix [N_anchors, D].
    """
    optimizer = torch.optim.Adam(anchor_selector.parameters(), lr=lr)
    for epoch in range(epochs):
        anchors = anchor_selector(embeddings)
        loss_cov = coverage_loss(anchors, embeddings)
        loss_div = diversity_loss(anchors, exponent=exp)
        loss = diversity_weight * loss_div + coverage_weight * loss_cov + anti_collapse_w * anti_collapse_loss(anchors)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.7f}, weighted coverage={loss_cov.item()*coverage_weight:.7f}, weighted diversity={loss_div.item()*diversity_weight:.4f}, weighted anti-collapse={anti_collapse_loss(anchors).item()*anti_collapse_w:.4f}")
    return anchor_selector(embeddings)

def get_optimized_anchors(emb, anchor_num, epochs=50, lr=1e-1,
                          coverage_weight=1.0, diversity_weight=1.0, anti_collapse_w=1.0, exponent=1, verbose=True, device='cpu'):
    """
    For a list of embeddings (numpy arrays), optimize anchors on the first run's embeddings
    and then compute the corresponding anchors for every run.
    
    Returns:
      anchor_selector: the trained AnchorSelector.
      P_anchors_list: list of anchors for each run.
    """
    if verbose:
      print("Optimizing P anchors...")
    # Optimize on the first run's embeddings
    X_first = emb[0]
    X_first_tensor = torch.from_numpy(X_first).to(device)
    anchor_selector = AnchorSelector(N=X_first_tensor.shape[0], N_anchors=anchor_num).to(device)
    optimize_anchors(anchor_selector, X_first_tensor, epochs=epochs, lr=lr,
                     coverage_weight=coverage_weight, diversity_weight=diversity_weight, anti_collapse_w=anti_collapse_w,
                     exp=exponent, verbose=verbose)
    
    # Compute anchors for each run using the optimized anchor_selector
    P_anchors_list = []
    for emb in emb:
        X_tensor_run = torch.from_numpy(emb).to(device)
        anchors_run = anchor_selector(X_tensor_run)
        P_anchors_list.append(anchors_run.cpu().detach().numpy())
    
    return anchor_selector, P_anchors_list

