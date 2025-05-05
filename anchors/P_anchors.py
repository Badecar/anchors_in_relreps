import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import torch.optim as optim

def optimize_weights(center, candidates, lambda_reg=0.1, lr=1e-2, epochs=200, device="cuda"):
    """
    Optimize weights on GPU so that a weighted combination of candidate points approximates center.
    The objective is:
      ||sum_i w_i * candidate_i - center||² + lambda_reg * KL(w || uniform)
    We enforce w_i >= 0 and sum_i w_i = 1 by representing the weights via softmax.
    
    Args:
      center: torch.Tensor of shape (D,)
      candidates: torch.Tensor of shape (n_candidates, D)
      lambda_reg: float, regularization coefficient.
      lr: learning rate
      epochs: number of optimization steps
      device: device string, e.g. "cuda" or "cpu"
    
    Returns:
      weights: numpy array of shape (n_candidates,), representing the optimized weights.
    """
    if not isinstance(center, torch.Tensor):
        center = torch.from_numpy(np.array(center))
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(np.array(candidates))
    center = center.to(device)
    candidates = candidates.to(device)
    
    n = candidates.shape[0]
    # We optimize an unconstrained parameter vector which will be normalized via softmax.
    params = torch.zeros(n, device=device, requires_grad=True)
    
    optimizer = optim.Adam([params], lr=lr)
    eps = 1e-8
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Apply softmax to ensure positivity and unity sum.
        w = torch.softmax(params, dim=0)
        reconstruction = torch.matmul(w, candidates)
        reconstruction_error = torch.norm(reconstruction - center) ** 2
        
        # Compute KL divergence with the uniform distribution (each weight ~ 1/n)
        kl = torch.sum(w * torch.log(w * n + eps))
        
        loss = reconstruction_error + lambda_reg * kl
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        final_weights = torch.softmax(params, dim=0).cpu().numpy()
    
    return final_weights

# def optimize_weights(center, candidates, lambda_reg=0.1):
#     """
#     Optimize weights so that a weighted combination of candidate points approximates the center
#     with an entropy-like regularization.
    
#     Args:
#       center: numpy array of shape [D,]
#       candidates: numpy array of shape [n_closest, D]
#       lambda_reg: regularization coefficient.
    
#     Returns:
#       weights: numpy array of shape [n_closest,]
#     """
#     n = candidates.shape[0]
#     epsilon = 1e-8
#     def objective(w):
#         reconstruction_error = np.linalg.norm(np.dot(w, candidates) - center)**2
#         # KL divergence with uniform distribution (ideal weight = 1/n)
#         kl = np.sum(w * np.log(w * n + epsilon))
#         return reconstruction_error + lambda_reg * kl
#     cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
#     bounds = [(0, None)] * n
#     w0 = np.ones(n) / n
#     res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
#     if res.success:
#         return res.x
#     else:
#         return w0

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
        # You can control the temperature in softmax for sharper assignment
        # Higher temperature -> softer assignment
        T = 0.2
        P = F.softmax(self.Q / T, dim=1)
        anchors = P @ X  # [N_anchors, D]
        return anchors


# NOTE: Eucl
def diversity_loss_eucl(anchors, exponent=1.0, scale=1.0/np.sqrt(2)):
    pdist_vals = torch.pdist(anchors, p=2)
    # scaled_pdist_vals = pdist_vals * scale # Only needed if we want to mix cossim and eucl
    return -torch.mean(pdist_vals ** exponent)  # negative so maximizing diversity

def coverage_loss_eucl(anchors, embeddings, scale=1.0/np.sqrt(2)):
    # (Here we use squared Euclidean distance)
    dists = torch.cdist(embeddings, anchors, p=2)  # [N, N_anchors]
    # scaled_dist = dists * scale # Only needed if we want to mix cossim and eucl
    min_dists, _ = torch.min(dists, dim=1)
    return torch.mean(min_dists)

#NOTE: Cosine similarity version of losses
def diversity_loss_cossim(anchors, exponent=1.0):
    # Compute pairwise distances (for example, cosine or Euclidean)
    anchors_norm = F.normalize(anchors, p=2, dim=1)
    sim_matrix = anchors_norm @ anchors_norm.t()
    idx = torch.triu_indices(sim_matrix.size(0), sim_matrix.size(1), offset=1)
    cosine_sim_values = abs(sim_matrix[idx[0], idx[1]])
    return torch.mean(cosine_sim_values ** exponent)

def coverage_loss_cossim(anchors, embeddings):
    # For each embedding, compute its distance to each anchor and take the minimum.
    anchors_norm = F.normalize(anchors, p=2, dim=1)
    emb_norm = F.normalize(embeddings, p=2, dim=1)
    sim = abs(emb_norm @ anchors_norm.t())
    min_dists, _ = torch.min(sim, dim=1)
    return -torch.mean(min_dists)

#NOTE: Using Mahalanobis distance.
def coverage_loss_mahalanobis(anchors, embeddings, inv_cov):
    """
    For each embedding, compute its Mahalanobis distance to each anchor and take the minimum.
    Returns the negative mean of these minimum distances.
    
    anchors: [N_anchors, D]
    embeddings: [N, D]
    inv_cov: [D, D] inverse covariance matrix.
    """
    # Compute differences: shape [N, N_anchors, D]
    diff = embeddings.unsqueeze(1) - anchors.unsqueeze(0)
    # Compute Mahalanobis distances: d_i,j = sqrt((x_i - a_j)^T * inv_cov * (x_i - a_j))
    dists = torch.sqrt(torch.einsum("nid,ij,njd->ni", diff, inv_cov, diff) + 1e-8)
    min_dists, _ = torch.min(dists, dim=1)
    return -torch.mean(min_dists)

def diversity_loss_mahalanobis(anchors, inv_cov, exponent=1.0):
    """
    Computes the average pairwise Mahalanobis distance (raised to the given exponent)
    between anchors as a measure of diversity.
    
    anchors: [N_anchors, D]
    inv_cov: [D, D] inverse covariance matrix.
    """
    n = anchors.size(0)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            diff = anchors[i] - anchors[j]
            # Mahalanobis distance: sqrt(diff^T * inv_cov * diff)
            dist = torch.sqrt((diff.unsqueeze(0) @ inv_cov @ diff.unsqueeze(1)).squeeze() + 1e-8)
            dists.append(dist)
    if len(dists) == 0:
        return torch.tensor(0.0, device=anchors.device)
    dists = torch.stack(dists)
    return -torch.mean(dists ** exponent)

# Batched version for Mahalanobis coverage loss.
def coverage_loss_mahalanobis_batched(anchors, embeddings, inv_cov, batch_size=2**16):
    total_loss = 0.0
    total_samples = 0
    for i in range(0, embeddings.size(0), batch_size):
        emb_batch = embeddings[i:i+batch_size]                         # shape: (B, D)
        diff = emb_batch.unsqueeze(1) - anchors.unsqueeze(0)             # (B, N_anchors, D)
        # print(f"got through the {i}th diff")
        # Using the same einsum string as in the test function:
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        min_dists, _ = torch.min(dists, dim=1)
        total_loss += torch.sum(min_dists)
        total_samples += emb_batch.size(0)
    return total_loss / total_samples

# Batched version for Mahalanobis diversity loss.
def diversity_loss_mahalanobis_batched(anchors, inv_cov, exponent=1.0, batch_size=2**16):
    n = anchors.size(0)
    total_loss = 0.0
    count = 0
    for i in range(0, n, batch_size):
        end_i = min(i+batch_size, n)
        for j in range(i+1, n, batch_size):
            end_j = min(j+batch_size, n)
            a_batch = anchors[i:end_i]   # shape: (B1, D)
            b_batch = anchors[j:end_j]   # shape: (B2, D)
            diff = a_batch.unsqueeze(1) - b_batch.unsqueeze(0)  # (B1, B2, D)
            dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8) # Dont save cov matrix in grad
            total_loss += torch.sum(dists ** exponent)
            count += dists.numel()
    if count == 0:
        return torch.tensor(0.0, device=anchors.device)
    return -total_loss / count


def anti_collapse_loss(anchors):
    # return torch.mean(torch.abs(1 - torch.norm(anchors, dim=1)))
    return torch.mean(torch.abs(1 - torch.norm(anchors, dim=1)))

def anchor_size_loss(anchors):
   return -torch.mean(torch.norm(anchors, dim=1))


def optimize_anchors(anchor_selector, embeddings, epochs=100, lr=1e-3, coverage_weight=1.0, diversity_weight=1.0, anti_collapse_w=1.0, exp=1, dist_measure="cosine", verbose=True, device='cpu'):
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
        if dist_measure == "euclidean":
            loss_cov = coverage_loss_eucl(anchors, embeddings)
            loss_div = diversity_loss_eucl(anchors, exponent=exp)
            anti_collapse_w = 0.0  # No anti-collapse loss needed for euclidean
        elif dist_measure == "cosine":  
            loss_cov = coverage_loss_cossim(anchors, embeddings)
            loss_div = diversity_loss_cossim(anchors, exponent=exp)
            anti_collapse_w = 0.0  # No anti-collapse loss needed for euclidean
        elif dist_measure == "mahalanobis":
            inv_cov = torch.linalg.inv(compute_covariance_matrix(embeddings) + 1e-6 * torch.eye(embeddings.size(1), device=embeddings.device)).detach()
            loss_cov = coverage_loss_mahalanobis_batched(anchors, embeddings, inv_cov)
            loss_div = diversity_loss_mahalanobis_batched(anchors, inv_cov, exponent=exp)
        else:
            raise ValueError(f"dist_measure must be one of 'euclidean', 'cosine', or 'mahalanobis' but was {dist_measure}.")
        loss = diversity_weight * loss_div + coverage_weight * loss_cov + anti_collapse_w * anchor_size_loss(anchors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.7f}, weighted coverage={loss_cov.item()*coverage_weight:.7f}, weighted diversity={loss_div.item()*diversity_weight:.7f}, weighted anti-collapse={anti_collapse_loss(anchors).item()*anti_collapse_w:.4f}")
    return anchor_selector(embeddings)


def get_optimized_anchors(emb, anchor_num, epochs=50, lr=1e-1,
                          coverage_weight=1.0, diversity_weight=1.0, anti_collapse_w=1.0, exponent=1, dist_measure="cosine", verbose=True, device='cpu'):
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
                     exp=exponent, dist_measure=dist_measure, verbose=verbose, device=device)
    
    # Compute anchors for each run using the optimized anchor_selector
    P_anchors_list = []
    for emb in emb:
        X_tensor_run = torch.from_numpy(emb).to(device)
        anchors_run = anchor_selector(X_tensor_run)
        P_anchors_list.append(anchors_run.cpu().detach().numpy())
    
    return anchor_selector, P_anchors_list

def get_P_anchors_clustered(emb, anchor_num, n_closest=20, lambda_reg=0.1, epochs=50, lr=1e-1,
                            coverage_weight=1.0, diversity_weight=1.0, anti_collapse_w=1.0, exponent=1,
                            dist_measure="cosine", verbose=True, device='cpu'):
    """
    Trains an AnchorSelector on the first embedding in emb and then refines its anchors
    similarly to KMeans clusters.
    
    Steps:
      1. Train the anchor selector on the first embedding.
      2. Compute the initial anchors for the first embedding.
      3. For each anchor (cluster center), find the n_closest datapoints (Euclidean) in the first embedding,
         and recompute the anchor as a weighted average using optimized weights.
      4. For every embedding in emb, compute refined anchors using the same candidate indices and weights.
    
    Args:
      emb (list of numpy arrays): each array is of shape [N, D].
      anchor_num (int): number of anchors.
      n_closest (int): number of candidate datapoints per anchor.
      lambda_reg (float): regularization for optimize_weights.
      epochs, lr, coverage_weight, diversity_weight, anti_collapse_w, exponent, dist_measure, verbose, device:
         parameters passed to get_optimized_anchors.
    
    Returns:
      anchor_selector: the trained AnchorSelector.
      P_anchors_list: list of refined anchors (each as a numpy array of shape [anchor_num, D]) for each embedding.
      clusters_info: list of tuples for each anchor: (candidate_indices, weights, refined_anchor)
    """
    # Train anchor selector on the first embedding
    X_first = emb[0]
    X_first_tensor = torch.from_numpy(X_first).to(device)
    anchor_selector, _ = get_optimized_anchors(emb=[X_first], anchor_num=anchor_num, epochs=epochs, lr=lr,
                                               coverage_weight=coverage_weight, diversity_weight=diversity_weight,
                                               anti_collapse_w=anti_collapse_w, exponent=exponent,
                                               dist_measure=dist_measure, verbose=verbose, device=device)
    # Compute initial anchors from the trained selector on the first embedding
    initial_anchors = anchor_selector(X_first_tensor)  # shape: [anchor_num, D]
    initial_anchors_np = initial_anchors.cpu().detach().numpy()

    clusters_info = []
    refined_anchors_first = []
    # For each anchor, find its nearest n_closest points and refine the anchor.
    for i in tqdm(range(anchor_num), desc="Refining Anchors"):
        anchor_center = initial_anchors_np[i]
        # Compute Euclidean distances from anchor_center to all datapoints in X_first
        dists = np.linalg.norm(X_first - anchor_center, axis=1)
        candidate_indices = np.argsort(dists)[:n_closest]
        candidate_points = X_first[candidate_indices]
        weights = optimize_weights(anchor_center, candidate_points, lambda_reg=lambda_reg)
        refined_anchor = np.average(candidate_points, axis=0, weights=weights)
        clusters_info.append((candidate_indices, weights, refined_anchor))
        refined_anchors_first.append(refined_anchor)
    refined_anchors_first = np.vstack(refined_anchors_first)  # shape: [anchor_num, D]
    
    # For each embedding in emb, compute refined anchors using the same candidate indices and weights
    P_anchors_list = []
    for X in emb:
        refined_anchors = []
        for candidate_indices, weights, _ in clusters_info:
            candidate_points = X[candidate_indices]
            refined_anchor = np.average(candidate_points, axis=0, weights=weights)
            refined_anchors.append(refined_anchor)
        refined_anchors = np.vstack(refined_anchors)
        P_anchors_list.append(refined_anchors)
    
    return anchor_selector, P_anchors_list, clusters_info