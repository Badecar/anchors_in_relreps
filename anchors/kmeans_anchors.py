
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize
from scipy.special import softmax
from .P_anchors import optimize_weights

import os
# To get around warning
os.environ["LOKY_MAX_CPU_COUNT"] = "8"

def get_kmeans_based_anchors(emb, idx_list, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
    """
    Computes anchors based on KMeans centers and the n_closest candidate points per cluster.
    
    Steps:
      1. Run KMeans on the first run's embeddings (emb[0]) with n_clusters=anchor_num.
      2. For each KMeans center, compute Euclidean distances to all points in emb[0],
         select the n_closest points and map to global indices using idx_list[0].
      3. Use constrained minimization (with an extra regularization term) to compute
         weights such that the weighted combination of these candidate points approximates the center.
      4. For each run, use these global indices to fetch candidate points and compute a weighted
         average using the optimized weights.
    
    Args:
      emb: list of numpy arrays, each of shape [N, D] (embeddings for one run).
      idx_list: list of arrays/lists mapping each embedding to a global dataset index.
      anchor_num: number of anchors (clusters) to compute.
      n_closest: number of closest points (candidates) to combine per cluster.
      kmeans_seed: random seed for KMeans.
      
    Returns:
      anchors_all_runs: list of numpy arrays, each of shape [anchor_num, D] representing anchors for each run.
      clusters_info: list of tuples for each cluster: (global_indices, weights, center)
    """
    # Use first run embeddings for clustering.
    X_first = emb[0]  # shape [N, D]
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(X_first)
    centers = kmeans.cluster_centers_  # shape [anchor_num, D]
    
    clusters_info = []  # store (global_indices, weights, center) for each cluster
    for i,center in enumerate(centers):
        dists = np.linalg.norm(X_first - center, axis=1)
        closest_idxs = np.argsort(dists)[:n_closest]
        global_idxs = np.array(idx_list[0])[closest_idxs]
        candidate_points = X_first[closest_idxs]
        weights = optimize_weights(center, candidate_points)
        clusters_info.append((global_idxs, weights, center))
        if i == 0 and verbose:
            print(weights)
    
    anchors_all_runs = []
    for run_id, (X, idx) in enumerate(zip(emb, idx_list)):
        anchors_run = []
        idx = np.array(idx)
        for global_idxs, weights, _ in clusters_info:
            positions = [np.where(idx == g)[0][0] for g in global_idxs if np.any(idx == g)]
            if len(positions) == 0:
                anchor = np.zeros(X.shape[1])
            else:
                points = X[positions]
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