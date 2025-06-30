
import numpy as np
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm.auto import tqdm
import os

# To get around warning
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

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

def get_kmeans_anchors(embeddings, anchor_num, idx_list=None, n_closest=20, kmeans_seed=42, verbose=False):
    """
    Computes anchors for a list of embeddings based on KMeans clustering applied to the first embedding.
    
    IMPORTANT: If idx_list is provided then each embedding's rows may be in a different order.
    The provided idx_list must be a list of np.ndarray such that for each embedding, its idx_list[i]
    contains the unique identifiers for each sample. The candidate indices from the base embedding are
    then used to lookup the corresponding rows in each embedding's idx_list.
    
    Parameters:
        embeddings (list of np.ndarray): List of embeddings arrays (each of shape [N_i, D_i]). The first embedding is used for clustering.
        anchor_num (int): Number of clusters/anchors.
        idx_list (list of np.ndarray, optional): List of index arrays corresponding to each embedding.
            If not provided, np.arange is used for the base embedding and the same ordering is assumed for all embeddings.
        n_closest (int): Number of closest points to a center used for weighted averaging.
        kmeans_seed (int): Seed for KMeans.
        verbose (bool): Whether to print debug info.
        
    Returns:
        anchors_list (list of np.ndarray): List of anchors arrays, one per input embedding.
        clusters_info (list): List of tuples for each cluster: (candidate_indices, weights, center)
    """
    # Use the first embedding for clustering.
    base_emb = embeddings[0]
    N = base_emb.shape[0]
    if idx_list is None:
        global_idx = np.arange(N)
    else:
        # Use provided index list for the base embedding.
        global_idx = idx_list[0]
    
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(base_emb)
    centers = kmeans.cluster_centers_
    
    clusters_info = []
    # Prepare a list to collect anchors per embedding.
    anchors_list = [[] for _ in embeddings]
    
    for center in tqdm(centers, desc="Processing clusters"):
        # Compute distances on the base embedding.
        dists = np.linalg.norm(base_emb - center, axis=1)
        candidate_order = np.argsort(dists)[:n_closest]
        candidate_global = global_idx[candidate_order]
        candidate_points = base_emb[candidate_order]
        
        # Compute weights for candidates.
        weights = optimize_weights(center, candidate_points)
        clusters_info.append((candidate_global, weights, center))
        
        # For each embedding, select the corresponding candidate rows.
        for i, emb in enumerate(embeddings):
            if idx_list is None:
                # Same ordering is assumed.
                candidate_positions = candidate_order
            else:
                # For embedding i, find positions where idx_list[i] matches each candidate in candidate_global.
                # This ensures that we are using the same points based on their unique id.
                candidate_positions = np.array([np.where(idx_list[i] == cand)[0][0] for cand in candidate_global])
            candidate_values = emb[candidate_positions]
            anchor_val = np.average(candidate_values, axis=0, weights=weights)
            anchors_list[i].append(anchor_val)
    
    # Convert list of anchors to a NumPy array for each embedding.
    anchors_list = [np.vstack(anchors) for anchors in anchors_list]
    
    if verbose:
        print("KMeans centers (first):", centers[0])
        print("First cluster candidate indices:", clusters_info[0][0])
        for i, anchors in enumerate(anchors_list):
            print(f"First approximated anchor for embedding {i}:", anchors[0])
    
    return anchors_list, clusters_info


def get_kmeans_based_anchors_softmax(emb, idx_list, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
    """
    Computes anchors based on KMeans centers and the 20 closest points to each center.

    Steps:
      1. Run KMeans on the first run's embeddings (emb[0]) with n_clusters=anchor_num.
      2. For each KMeans center, compute Euclidean distances to all points in emb[0].
         Then select the n_closest points and map their positions to global indices using idx_list[0].
      3. Compute weights via a softmax on the negative distances so that closer points have higher weight.
         (This weighting approximates the linear combination that equals the cluster center.)
      4. For each run, use these global indices to find the corresponding points in that run's embeddings
         (using idx_list) and compute a weighted average of their embeddings.
         
    Additionally, for the first embedding run, prints the actual KMeans centers and the approximated anchors.
    
    Args:
      emb: list of numpy arrays, each of shape [N, D] containing the embeddings for one run.
      idx_list: list of arrays/lists mapping each embedding to a global dataset index.
      anchor_num: number of anchors (clusters) to compute.
      n_closest: number of closest points to combine per cluster.
      kmeans_seed: random seed for KMeans.
      
    Returns:
      anchors_all_runs: list of numpy arrays, each of shape [anchor_num, D] representing anchors for each run.
      clusters_info: list of tuples for each anchor from the first run: (global_indices, weights, center)
    """
    # Use first run embeddings for clustering.
    X_first = emb[0]  # shape [N, D]
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(X_first)
    centers = kmeans.cluster_centers_  # shape [anchor_num, D]
    
    clusters_info = []  # will store (global_indices, weights, center) for each cluster
    for center in centers:
        # Compute Euclidean distances from the center to all points in X_first.
        dists = np.linalg.norm(X_first - center, axis=1)
        # Get indices of n_closest points in the first run
        closest_idxs = np.argsort(dists)[:n_closest]
        # Map these indices to the actual global indices using idx_list[0]
        global_idxs = np.array(idx_list[0])[closest_idxs]
        # Compute weights using a softmax over the negative distances so that closer points count more.
        selected_dists = dists[closest_idxs]
        weights = softmax(-selected_dists)  # shape (n_closest,)
        clusters_info.append((global_idxs, weights, center))
    
    anchors_all_runs = []
    first_run_approximations = None  # to store approximated anchors for the first run
    for run_id, (X, idx) in enumerate(zip(emb, idx_list)):
        anchors_run = []
        idx = np.array(idx)
        for global_idxs, weights, _ in clusters_info:
            # Find the positions in the current run that correspond to each global index.
            positions = [np.where(idx == g)[0][0] for g in global_idxs if np.any(idx == g)]
            if len(positions) == 0:
                # Fallback: if none of the global indices are found in this run, use a zero vector.
                anchor = np.zeros(X.shape[1])
            else:
                points = X[positions]  # shape: [n_found, D]
                # In case not all n_closest points are found, adjust the weights accordingly.
                if len(positions) < len(weights):
                    weights_adjusted = weights[:len(positions)]
                    weights_adjusted = weights_adjusted / np.sum(weights_adjusted)
                else:
                    weights_adjusted = weights
                anchor = np.average(points, axis=0, weights=weights_adjusted)
            anchors_run.append(anchor)
        anchors_run = np.vstack(anchors_run)
        anchors_all_runs.append(anchors_run)
        
        # For the first run only: print the actual KMeans centers and the approximated anchors.
        if run_id == 0 and verbose:
            print("KMeans centers:")
            for center in centers[:1]:
                print(center)
            print("Approximated anchors from points (weighted average):")
            for anchor in anchors_run[:1]:
                print(anchor)
                
    return anchors_all_runs, clusters_info