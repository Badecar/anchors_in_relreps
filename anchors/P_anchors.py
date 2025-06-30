import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from anchors.kmeans_anchors import optimize_weights
from relreps import compute_covariance_matrix

TEMP = 0.2

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
        T = TEMP
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
    # Compute differences: shape [N, N_anchors, D]
    diff = embeddings.unsqueeze(1) - anchors.unsqueeze(0)
    # Compute Mahalanobis distances: d_i,j = sqrt((x_i - a_j)^T * inv_cov * (x_i - a_j))
    dists = torch.sqrt(torch.einsum("nid,ij,njd->ni", diff, inv_cov, diff) + 1e-8)
    min_dists, _ = torch.min(dists, dim=1)
    return -torch.mean(min_dists)

def diversity_loss_mahalanobis(anchors, inv_cov, exponent=1.0):
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

def anti_collapse_loss(anchors):
    # return torch.mean(torch.abs(1 - torch.norm(anchors, dim=1)))
    return torch.mean(torch.abs(1 - torch.norm(anchors, dim=1)))

# def anchor_size_loss(anchors):
#    return -torch.mean(torch.norm(anchors, dim=1))

def anchor_size_loss(anchors, target=1.0):
    # Penalize when anchors have norms below the target.
    current_sizes = torch.norm(anchors, dim=1)
    loss = torch.mean(torch.abs(target - current_sizes) ** 2)
    return loss

def parameter_magnitude_loss(anchor_selector):
    # Penalize for large magnitude in the parameter matrix P.
    return torch.norm(anchor_selector.P, p=2)

def clustering_loss(anchors, X, P, gamma=0.1):
    """
    Compute a clustering loss using the assignment weights.
    
    anchors: [N_anchors, D] computed as P @ X.
    X: [N, D] embeddings.
    P: [N_anchors, N] soft-assignment weights.
    gamma: coefficient that scales the entropy term.
    
    Returns:
      loss: a scalar tensor. For each anchor, we penalize high weighted distance
            and reward high entropy.
    """
    # Compute the Euclidean distances from each anchor to each point in X.
    dists = torch.cdist(anchors, X, p=2)  # shape: [N_anchors, N]
    # Weighted distance term: penalize anchors that put mass on faraway points.
    loss_distance = torch.sum(P * dists) / P.shape[0]
    
    # Entropy term: discourage the selector from focusing on only one or two points.
    entropy = -torch.sum(P * torch.log(P + 1e-8)) / P.shape[0]
    
    # We subtract the entropy term so that higher entropy (more spread-out assignments)
    # reduce the loss.
    return loss_distance - gamma * entropy
    # return gamma*entropy

def locality_kl_loss(anchors, X, P, sigma=1.0):
    """
    Compute a KL divergence loss encouraging the soft assignments P to be close to a Gaussian target.
    
    anchors: [N_anchors, D] computed as P @ X.
    X: [N, D] embeddings.
    P: [N_anchors, N] soft-assignment weights.
    sigma: standard deviation for the Gaussian target.
    
    Returns:
      A scalar tensor representing the KL divergence loss.
    """
    # Compute distances from each anchor to all points: [N_anchors, N]
    dists = torch.cdist(anchors, X, p=2)
    # Compute the target Gaussian distribution for each anchor
    # exp(-d^2/(2*sigma^2)) then normalize over points.
    target = torch.exp(- dists ** 2 / (2 * sigma ** 2))
    target = target / (target.sum(dim=1, keepdim=True) + 1e-8)
    
    # Compute KL divergence: target * log(target/(P+eps))
    eps = 1e-8
    kl = torch.sum(target * torch.log((target + eps) / (P + eps)), dim=1)
    # Average over the anchors
    return kl.mean()

def optimize_anchors(anchor_selector, embeddings, temp=0.2, epochs=100, lr=1e-3, coverage_weight=1.0, diversity_weight=1.0, anti_collapse_w=1.0, exp=1, dist_measure="cosine", verbose=True, device='cpu'):
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
    for epoch in tqdm(range(epochs)):
        anchors = anchor_selector(embeddings)
        P = torch.softmax(anchor_selector.Q / TEMP, dim=1)
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
            loss_cov = coverage_loss_mahalanobis(anchors, embeddings, inv_cov)
            loss_div = diversity_loss_mahalanobis(anchors, inv_cov, exponent=exp)
        else:
            raise ValueError(f"dist_measure must be one of 'euclidean', 'cosine', or 'mahalanobis' but was {dist_measure}.")
        #Getting decent results (better than basic P) with cl loss and high gamma.
        #But still having issues with the weights concentrated in one or two values (also a slight issue with kmeans atm)
        #Maybe look into using the same optimization as kmeans or setting n_closest value or the like
        #These values
        cl_w = 0.4
        lc_w = 20
        cl_loss = clustering_loss(anchors, embeddings, P, gamma=10)
        locality_loss = locality_kl_loss(anchors, embeddings, P, sigma=0.1)
        loss = (diversity_weight * loss_div +
                coverage_weight * loss_cov +
                cl_w * cl_loss +
                lc_w * locality_loss +
                anti_collapse_w * anchor_size_loss(anchors))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, weighted coverage={loss_cov.item()*coverage_weight:.4f}, weighted diversity={loss_div.item()*diversity_weight:.4f}, weighted clusterweight={cl_loss.item()*cl_w:.4f}, weighted localityloss={locality_loss.item()*lc_w:.4f}, weighted anti-collapse={anti_collapse_loss(anchors).item()*anti_collapse_w:.4f}")
    
    anchors = anchor_selector(embeddings)
    return anchors

def get_P_anchors(emb, anchor_num, clustered=False, n_closest=20, lambda_reg=0.1,
                    epochs=50, lr=1e-1, coverage_weight=1.0, diversity_weight=None,
                    anti_collapse_w=1.0, exponent=1, dist_measure="cosine", verbose=True, device='cpu'):
    """
    For a list of embeddings (numpy arrays), trains an AnchorSelector on the first embedding.
    If clustered is False, computes anchors for each run.
    If clustered is True, refines each anchor by finding its nearest n_closest points in the first embedding,
    optimizing weights to recompute the anchor, and then applies the candidate indices and weights to every run.
    
    Args:
        emb (list of numpy arrays): each array is of shape [N, D].
        anchor_num (int): number of anchors.
        clustered (bool): If True, refine anchors using nearest neighbor optimization similarly to KMeans.
        n_closest (int): Number of candidate datapoints per anchor (used when clustered=True).
        lambda_reg (float): Regularization coefficient for optimize_weights.
        epochs (int): Number of epochs for training the AnchorSelector.
        lr (float): Learning rate.
        coverage_weight (float): Weight for the coverage loss.
        diversity_weight (float): Weight for the diversity loss.
        anti_collapse_w (float): Weight for the anchor size loss.
        exponent (int): Exponent used in diversity loss.
        dist_measure (str): Distance measure, one of "euclidean", "cosine", or "mahalanobis".
        verbose (bool): If True, print progress.
        device (str): Device string, e.g. "cpu" or "cuda".
    
    Returns:
        If clustered is False:
            anchor_selector: The trained AnchorSelector.
            P_anchors_list: List of anchors (each as a numpy array of shape [anchor_num, D]) for each embedding.
        If clustered is True:
            anchor_selector: The trained AnchorSelector.
            P_anchors_list: List of refined anchors for each embedding.
            clusters_info: List of tuples for each anchor: (candidate_indices, weights, refined_anchor)
    """
    if diversity_weight is None:
        diversity_weight = 1.0 - coverage_weight
    if verbose:
        print("Training AnchorSelector...")
    # Optimize on the first run's embeddings
    X_first = emb[0]
    X_first_tensor = torch.from_numpy(X_first).to(device)
    anchor_selector = AnchorSelector(N=X_first_tensor.shape[0], N_anchors=anchor_num).to(device)
    optimize_anchors(anchor_selector, X_first_tensor, epochs=epochs, lr=lr,
                        coverage_weight=coverage_weight, diversity_weight=diversity_weight,
                        anti_collapse_w=anti_collapse_w, exp=exponent, dist_measure=dist_measure,
                        verbose=verbose, device=device)
    
    # If not clustered, simply compute anchors for every embedding
    if not clustered:
        P_anchors_list = []
        for emb_arr in emb:
            X_tensor_run = torch.from_numpy(emb_arr).to(device)
            anchors_run = anchor_selector(X_tensor_run)
            P_anchors_list.append(anchors_run.cpu().detach().numpy())
        if verbose:
            plot_anchor_histogram(anchor_selector, X_first_tensor, temp=0.2, sample_idx=0)
        return anchor_selector, P_anchors_list, []

    # IF CLUSTERED PERFORM CLUSTERING ALGORITHM #

    if verbose:
        print("Refining anchors using clustering...")
    # Compute initial anchors on the first embedding
    initial_anchors = anchor_selector(X_first_tensor)  # shape: [anchor_num, D]
    initial_anchors_np = initial_anchors.cpu().detach().numpy()
    
    clusters_info = []
    refined_anchors_first = []
    # For each anchor, find its n_closest points in the first embedding,
    # and use optimized weights to compute a refined anchor.
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
    
    # For each embedding in emb, compute refined anchors using the same candidate indices and weights.
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

def plot_anchor_histogram(anchor_selector, X_first_tensor, temp=0.2, sample_idx=0):
    """
    Plots histograms of average weights vs Euclidean distance and prints the top 15 largest weights.
    
    Args:
      anchor_selector: The trained AnchorSelector containing parameter Q.
      X_first_tensor (Tensor): The tensor of embeddings from the first run.
      temp (float): Temperature used for softmax.
      sample_idx (int): Index of the sample anchor to plot.
    """
    with torch.no_grad():
        T = temp
        P = F.softmax(anchor_selector.Q / T, dim=1)
        anchors_out = P @ X_first_tensor
        sample_weights = P[sample_idx]
        sample_anchor = anchors_out[sample_idx].unsqueeze(0)  # shape: [1, D]
        sample_dists = torch.cdist(sample_anchor, X_first_tensor, p=2).squeeze(0)
        
        if sample_weights is not None and sample_dists is not None:
            sample_weights_np = sample_weights.cpu().detach().numpy()
            sample_dists_np = sample_dists.cpu().detach().numpy()
            
            # Parameters for adjustable binning
            overall_bins = 80          # for the overall histogram
            first_bins_count = 15      # number of smallest bins to zoom in on
            zoom_bins = 100            # new number of bins for the zoom-in region
            
            # Compute overall binned statistic (average weight per bin)
            bin_means, bin_edges, _ = binned_statistic(sample_dists_np, sample_weights_np,
                                                        statistic='mean', bins=overall_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
            
            # Find the cutoff for the "first" bins (using the upper edge of the first_bins_count bins)
            cutoff = bin_edges[first_bins_count]
            # Recompute binned statistics for the zoom region (samples with distance <= cutoff)
            zoom_mask = sample_dists_np <= cutoff
            if np.sum(zoom_mask) > 0:
                zoom_dists = sample_dists_np[zoom_mask]
                zoom_weights = sample_weights_np[zoom_mask]
                zoom_bin_edges = np.linspace(bin_edges[0], cutoff, zoom_bins+1)
                zoom_bin_means, _, _ = binned_statistic(zoom_dists, zoom_weights,
                                                        statistic='mean', bins=zoom_bin_edges)
                zoom_bin_centers = (zoom_bin_edges[:-1] + zoom_bin_edges[1:]) / 2.
            else:
                zoom_bin_means, zoom_bin_centers = None, None
            
            # Print the 15 largest weights for the anchor
            sorted_weights = sorted(sample_weights_np, reverse=True)
            top_15 = sorted_weights[:15]
            print("\nTop 15 largest weights for anchor {}:".format(sample_idx))
            for i, w in enumerate(top_15):
                print("{}. {:.15f}".format(i+1, w))
            
            # Save the overall histogram-like average weights plot
            plt.figure(figsize=(8, 4))
            plt.plot(bin_centers, bin_means, marker='o', linestyle='-')
            plt.xlabel("Euclidean Distance")
            plt.ylabel("Average Weight")
            plt.title("Anchor {}: Full Average Weight vs. Distance Histogram".format(sample_idx))
            plt.grid(True)
            plt.savefig("full_histogram_anchor_{}_plot.png".format(sample_idx))
            plt.close()
            
            # Save the zoom-in histogram plot if available
            if zoom_bin_means is not None:
                plt.figure(figsize=(8, 4))
                plt.plot(zoom_bin_centers, zoom_bin_means, marker='o', linestyle='-')
                plt.xlabel("Euclidean Distance")
                plt.ylabel("Average Weight")
                plt.title("Anchor {}: Zoomed Average Weight (first {} bins re-binned into {} bins)".format(sample_idx, first_bins_count, zoom_bins))
                plt.grid(True)
                plt.savefig("zoom_histogram_anchor_{}_plot.png".format(sample_idx))
                plt.close()