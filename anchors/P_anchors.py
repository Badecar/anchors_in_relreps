import torch
import torch.nn as nn
import torch.nn.functional as F

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


def diversity_loss(anchors, exponent=0.5):
    # anchors: [N_anchors, D]
    # Compute pairwise distances (for example, cosine or Euclidean)
    # and encourage anchors to be far apart.
    pdist_vals = torch.pdist(anchors, p=2)  # example with Euclidean
    return -torch.mean(pdist_vals ** exponent)  # negative so maximizing diversity


def coverage_loss(anchors, embeddings):
    # embeddings: [N, D] and anchors: [N_anchors, D]
    # For each embedding, compute its distance to each anchor and take the minimum.
    # Then, average over all embeddings.
    # (Here we use squared Euclidean distance)
    dists = torch.cdist(embeddings, anchors, p=2)  # [N, N_anchors]
    min_dists, _ = torch.min(dists, dim=1)
    return torch.mean(min_dists)

#NOTE: Cosine similarity version of losses
# def diversity_loss(anchors, exponent=0.5):
#     # anchors: [N_anchors, D]
#     # Normalize anchors so that cosine similarity can be computed.
#     anchors_norm = F.normalize(anchors, p=2, dim=1)
#     # Compute the cosine similarity matrix.
#     sim_matrix = anchors_norm @ anchors_norm.t()
#     # Get the upper triangular part without the diagonal.
#     idx = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
#     cosine_sim_values = sim_matrix[idx[0], idx[1]]
#     # Convert similarity to cosine distance.
#     cosine_distance = 1 - cosine_sim_values
#     # Negative so that maximizing distance reduces the loss.
#     return -torch.mean(cosine_distance ** exponent)


# def coverage_loss(anchors, embeddings):
#     # Normalize embeddings and anchors.
#     anchors_norm = F.normalize(anchors, p=2, dim=1)
#     emb_norm = F.normalize(embeddings, p=2, dim=1)
#     # Compute cosine similarity between each embedding and each anchor.
#     sim = emb_norm @ anchors_norm.t()
#     # Convert similarity to cosine distance.
#     cosine_distance = 1 - sim  # lower distance means higher similarity
#     # For each embedding, take the minimum distance (best matching anchor).
#     min_dists, _ = torch.min(cosine_distance, dim=1)
#     return torch.mean(min_dists)


def optimize_anchors(anchor_selector, embeddings, epochs=100, lr=1e-3, coverage_weight=1.0, diversity_weight=1.0, exp=1, verbose=True):
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
        loss = diversity_weight * loss_div + coverage_weight * loss_cov
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, weighted coverage={loss_cov.item()*coverage_weight:.4f}, weighted diversity={loss_div.item()*diversity_weight:.4f}")
    return anchor_selector(embeddings)

def get_optimized_anchors(emb, anchor_num, epochs=50, lr=1e-1,
                          coverage_weight=1.0, diversity_weight=1.0, exponent=1, verbose=True, device='cpu'):
    """
    For a list of embeddings (numpy arrays), optimize anchors on the first run's embeddings
    and then compute the corresponding anchors for every run.
    
    Returns:
      anchor_selector: the trained AnchorSelector.
      P_anchors_list: list of anchors for each run.
    """
    print("Optimizing P anchors...")
    # Optimize on the first run's embeddings
    X_first = emb[0]
    X_first_tensor = torch.from_numpy(X_first).to(device)
    anchor_selector = AnchorSelector(N=X_first_tensor.shape[0], N_anchors=anchor_num).to(device)
    optimize_anchors(anchor_selector, X_first_tensor, epochs=epochs, lr=lr,
                     coverage_weight=coverage_weight, diversity_weight=diversity_weight,
                     exp=exponent, verbose=verbose)
    
    # Compute anchors for each run using the optimized anchor_selector
    P_anchors_list = []
    for emb in emb:
        X_tensor_run = torch.from_numpy(emb).to(device)
        anchors_run = anchor_selector(X_tensor_run)
        P_anchors_list.append(anchors_run.cpu().detach().numpy())
    
    return anchor_selector, P_anchors_list

