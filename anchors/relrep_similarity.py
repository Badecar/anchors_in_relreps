import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm  # added progress bar support
from P_anchors import optimize_weights

# ------------------------ Utility functions ------------------------

def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_covariance_matrix(features):
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    cov = (centered.t() @ centered) / (features.size(0) - 1)
    return cov

def relative_projection_mahalanobis_batched(x, anchors, inv_cov, batch_size=512):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    result = []
    total = x.size(0)
    for i in tqdm(range(0, total, batch_size), desc="Mahalanobis projection", leave=False):
        x_batch = x[i:i+batch_size]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)
        # Negative distances so that higher means closer.
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        result.append(-dists)
    return torch.cat(result, dim=0)

def relative_projection_cosine(x, anchors):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    x_norm = F.normalize(x, p=2, dim=-1)
    anchors_norm = F.normalize(anchors, p=2, dim=-1).to(x.device)
    # Dot-product similarity: higher is more similar.
    return torch.einsum("bm,am->ba", x_norm, anchors_norm)

# def get_kmeans_anchors_clustered(src_emb, tgt_emb, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
#     # src_emb and tgt_emb are numpy arrays.
#     N = src_emb.shape[0]
#     global_idx = np.arange(N)
#     kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
#     kmeans.fit(src_emb)
#     centers = kmeans.cluster_centers_
    
#     clusters_info = []
#     anchors_src = []
#     anchors_tgt = []
    
#     for center in tqdm(centers, desc="Extracting KMeans anchors"):
#         dists = np.linalg.norm(src_emb - center, axis=1)
#         candidate_order = np.argsort(dists)[:n_closest]
#         candidate_global = global_idx[candidate_order]
#         candidate_points = src_emb[candidate_order]
#         # Optimize weights: here we use a simple inverse distance weighting.
#         weights = 1 / (dists[candidate_order] + 1e-8)
#         weights = weights / weights.sum()
#         clusters_info.append((candidate_global, weights, center))
#         anchor_src = np.average(src_emb[candidate_global], axis=0, weights=weights)
#         anchor_tgt = np.average(tgt_emb[candidate_global], axis=0, weights=weights)
#         anchors_src.append(anchor_src)
#         anchors_tgt.append(anchor_tgt)
    
#     anchors_src = np.vstack(anchors_src)
#     anchors_tgt = np.vstack(anchors_tgt)
    
#     if verbose:
#         print("First anchor (src):", anchors_src[0])
#         print("First anchor (tgt):", anchors_tgt[0])
    
#     return anchors_src, anchors_tgt, clusters_info


def get_kmeans_anchors_clustered(src_emb, tgt_emb, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
    # src_emb and tgt_emb are numpy arrays.
    N = src_emb.shape[0]
    global_idx = np.arange(N)
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(src_emb)
    centers = kmeans.cluster_centers_
    
    clusters_info = []
    anchors_src = []
    anchors_tgt = []
    
    for center in tqdm(centers, desc="Extracting KMeans anchors"):
        dists = np.linalg.norm(src_emb - center, axis=1)
        candidate_order = np.argsort(dists)[:n_closest]
        candidate_global = global_idx[candidate_order]
        candidate_points = src_emb[candidate_order]
        # Use the optimize_weights routine from P_anchors.
        weights = optimize_weights(center, candidate_points)
        clusters_info.append((candidate_global, weights, center))
        anchor_src = np.average(src_emb[candidate_global], axis=0, weights=weights)
        anchor_tgt = np.average(tgt_emb[candidate_global], axis=0, weights=weights)
        anchors_src.append(anchor_src)
        anchors_tgt.append(anchor_tgt)
    
    anchors_src = np.vstack(anchors_src)
    anchors_tgt = np.vstack(anchors_tgt)
    
    if verbose:
        print("First anchor (src):", anchors_src[0])
        print("First anchor (tgt):", anchors_tgt[0])
    
    return anchors_src, anchors_tgt, clusters_info

def normalize_np(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm==0] = 1
    return x / norm

# ------------------------ Evaluation Metrics ------------------------
def compute_mrr_cross(source, target, top_k=20, metric="cosine"):
    """
    For each sample i in source, compute the Mean Reciprocal Rank (MRR) between
    source and target using a vectorized cosine-similarity approach.
    This function always uses the same vectorized procedure as compute_latent_similarity.
    """
    # Normalize the embeddings rowwise.
    source_norm = source / np.linalg.norm(source, axis=1, keepdims=True)
    target_norm = target / np.linalg.norm(target, axis=1, keepdims=True)
    # Compute the similarity matrix (shape: [N, N])
    sim_matrix = np.dot(source_norm, target_norm.T)
    # For each query row, the true similarity is at the diagonal.
    true_sims = np.diag(sim_matrix).reshape(-1, 1)
    # Rank for each query: count how many scores in its row exceed its true similarity,
    # then add one (1-indexed rank)
    ranks = (sim_matrix > true_sims).sum(axis=1) + 1
    mrr = np.mean(1.0 / ranks)
    return mrr


def compute_jaccard_between(source, target, top_k=20, metric="cosine"):
    """
    For each sample i, compute the Jaccard similarity between the top_k neighbors 
    (excluding itself) in source and target. This function uses the vectorized similarity 
    approach (as in compute_latent_similarity) to extract sorted neighbor indices.
    """
    # Normalize the embeddings.
    source_norm = source / np.linalg.norm(source, axis=1, keepdims=True)
    target_norm = target / np.linalg.norm(target, axis=1, keepdims=True)
    # Compute similarity matrices for source and target.
    sim_source = np.dot(source_norm, source_norm.T)
    sim_target = np.dot(target_norm, target_norm.T)
    
    # For each sample, obtain the top_k neighbor indices (excluding self, which is at index 0 after sorting).
    all_jaccard = []
    for i in range(source.shape[0]):
        # argsort in descending order; skip index 0 (self)
        top_source = np.argsort(-sim_source[i])[1:top_k+1]
        top_target = np.argsort(-sim_target[i])[1:top_k+1]
        set_source = set(top_source)
        set_target = set(top_target)
        inter = len(set_source.intersection(set_target))
        union = len(set_source.union(set_target))
        all_jaccard.append(inter / union if union > 0 else 0)
    return np.mean(all_jaccard)

# ------------------------ PCA Plotting ------------------------

def plot_pca_scatter(embeddings, labels, classes, title):
    """
    embeddings: np.array shape (N, d)
    labels: np.array shape (N,)
    classes: list of classes to plot (only samples with these classes)
    """
    pca = PCA(n_components=2)
    proj = pca.fit_transform(embeddings)
    plt.title(title)
    for cl in classes:
        idx = (labels==cl)
        plt.scatter(proj[idx,0], proj[idx,1], label=f"Class {cl}", s=10)
    plt.legend()

# ------------------------ Main Experiment ------------------------

def main():
    print("Setting seeds and selecting device...")
    set_random_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading precomputed features...")
    current_path = os.path.dirname(os.path.abspath(__file__))
    features_file = os.path.join(current_path, "features_dict_CIFAR100_coarse.pt")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Please run the feature extraction script first.")
    features_dict = torch.load(features_file)

    # Select the two transformers.
    base_key = "vit_base_resnet50_384"
    target_key = "vit_small_patch16_224"
    if base_key not in features_dict or target_key not in features_dict:
        raise ValueError("One or both of the specified transformers are not available in the features dictionary.")

    print("Preparing evaluation data...")
    base_test_feats = features_dict[base_key]["test_features"]
    base_test_labels = features_dict[base_key]["test_labels"]
    target_test_feats = features_dict[target_key]["test_features"]
    target_test_labels = features_dict[target_key]["test_labels"]

    base_abs = base_test_feats.cpu().numpy()
    target_abs = target_test_feats.cpu().numpy()
    labels_abs = base_test_labels.cpu().numpy()  # assume labels are the same for both

    # --- Compute relative representations ---
    print("Computing relative representations...")

    anchor_num = 768  # you can adjust the number of anchors
    n_closest = 100
    kmeans_seed = 42

    print("Extracting kmeans+mahalanobis anchors...")
    anchors_base_np, anchors_target_np, _ = get_kmeans_anchors_clustered(
        src_emb=base_abs,
        tgt_emb=target_abs,
        anchor_num=anchor_num,
        n_closest=n_closest,
        kmeans_seed=kmeans_seed,
        verbose=False
    )
    anchors_base = torch.tensor(anchors_base_np, device=device, dtype=base_test_feats.dtype)
    anchors_target = torch.tensor(anchors_target_np, device=device, dtype=target_test_feats.dtype)
    
    print("Projecting base features with Mahalanobis distance...")
    cov_base = compute_covariance_matrix(base_test_feats.to(device))
    inv_cov_base = torch.linalg.inv(cov_base + 1e-6 * torch.eye(cov_base.size(0), device=cov_base.device))
    rel_base_km = relative_projection_mahalanobis_batched(base_test_feats.to(device), anchors_base, inv_cov_base)
    rel_base_km = rel_base_km.cpu().numpy()
    
    print("Projecting target features with Mahalanobis distance...")
    cov_target = compute_covariance_matrix(target_test_feats.to(device))
    inv_cov_target = torch.linalg.inv(cov_target + 1e-6 * torch.eye(cov_target.size(0), device=cov_target.device))
    rel_target_km = relative_projection_mahalanobis_batched(target_test_feats.to(device), anchors_target, inv_cov_target)
    rel_target_km = rel_target_km.cpu().numpy()

    print("Computing relative representations using random anchors with cosine similarity...")
    np.random.seed(kmeans_seed)
    rand_indices = np.sort(np.random.choice(base_abs.shape[0], anchor_num, replace=False))
    anchors_base_rand = base_abs[rand_indices]
    anchors_target_rand = target_abs[rand_indices]
    
    base_tensor = base_test_feats.to(device)
    target_tensor = target_test_feats.to(device)
    rel_base_rand = relative_projection_cosine(base_tensor, anchors_base_rand).T.cpu().numpy()
    rel_target_rand = relative_projection_cosine(target_tensor, anchors_target_rand).T.cpu().numpy()

    # --- Compute evaluation metrics ---
    print("Computing evaluation metrics...")
    top_k = 20

    # Absolute spaces evaluation using Euclidean.
    # print("Evaluating absolute representations (Euclidean)...")
    # mrr_abs = compute_mrr_cross(base_abs, target_abs, top_k=top_k, metric="cosine")
    # jaccard_abs = compute_jaccard_between(base_abs, target_abs, top_k=top_k, metric="cosine")

    # Relative representations (kmeans+mahalanobis) using Euclidean.
    print("Evaluating relative representations (kmeans+mahalanobis) (Euclidean)...")
    mrr_rel_km = compute_mrr_cross(rel_base_km, rel_target_km, top_k=top_k, metric="euclidean")
    jaccard_rel_km = compute_jaccard_between(rel_base_km, rel_target_km, top_k=top_k, metric="euclidean")

    # Relative representations (random+cossim) using cosine.
    print("Evaluating relative representations (random+cossim) (Cosine)...")
    mrr_rel_rand = compute_mrr_cross(rel_base_rand, rel_target_rand, top_k=top_k, metric="cosine")
    jaccard_rel_rand = compute_jaccard_between(rel_base_rand, rel_target_rand, top_k=top_k, metric="cosine")
    
    results_csv = os.path.join(current_path, "results_comparison.csv")
    with open(results_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Representation", "MRR", "Jaccard"])
        # writer.writerow(["absolute", mrr_abs, jaccard_abs])
        writer.writerow(["relative_kmeans", mrr_rel_km, jaccard_rel_km])
        writer.writerow(["relative_random", mrr_rel_rand, jaccard_rel_rand])
    print(f"Saved results to {results_csv}")

    # --- PCA Plots for 5 selected classes ---
    print("Generating PCA plots...")
    unique_classes = np.unique(labels_abs)
    selected_classes = unique_classes[:5]
    sel_idx = np.isin(labels_abs, selected_classes)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    pca_abs_base = PCA(n_components=2).fit_transform(base_abs[sel_idx])
    pca_abs_target = PCA(n_components=2).fit_transform(target_abs[sel_idx])

    axs[0,0].set_title(f"Absolute - {base_key}")
    for cl in selected_classes:
        cl_idx = np.where((labels_abs[sel_idx]==cl))[0]
        axs[0,0].scatter(pca_abs_base[cl_idx,0], pca_abs_base[cl_idx,1], s=10, label=f"Class {cl}")
    axs[0,0].legend()

    axs[0,1].set_title(f"Absolute - {target_key}")
    for cl in selected_classes:
        cl_idx = np.where((labels_abs[sel_idx]==cl))[0]
        axs[0,1].scatter(pca_abs_target[cl_idx,0], pca_abs_target[cl_idx,1], s=10, label=f"Class {cl}")
    axs[0,1].legend()

    pca_rel_base = PCA(n_components=2).fit_transform(rel_base_km[sel_idx])
    pca_rel_target = PCA(n_components=2).fit_transform(rel_target_km[sel_idx])

    axs[1,0].set_title(f"Relative (kmeans+mahalanobis) - {base_key}")
    for cl in selected_classes:
        cl_idx = np.where((labels_abs[sel_idx]==cl))[0]
        axs[1,0].scatter(pca_rel_base[cl_idx,0], pca_rel_base[cl_idx,1], s=10, label=f"Class {cl}")
    axs[1,0].legend()

    axs[1,1].set_title(f"Relative (kmeans+mahalanobis) - {target_key}")
    for cl in selected_classes:
        cl_idx = np.where((labels_abs[sel_idx]==cl))[0]
        axs[1,1].scatter(pca_rel_target[cl_idx,0], pca_rel_target[cl_idx,1], s=10, label=f"Class {cl}")
    axs[1,1].legend()

    plt.tight_layout()
    pca_plot_file = os.path.join(current_path, "pca_comparison.png")
    plt.savefig(pca_plot_file, dpi=300)
    print(f"Saved PCA plot to {pca_plot_file}")

if __name__ == "__main__":
    main()