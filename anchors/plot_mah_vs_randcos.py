import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import random
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import timm
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import minimize

from utils import set_random_seeds
from P_anchors import get_optimized_anchors, optimize_weights, get_P_anchors_clustered

os.environ["LOKY_MAX_CPU_COUNT"] = "8"

# Lambda module for wrapping lambda functions in nn.Sequential
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

# ----- Relative projection functions -----
def relative_projection_cosine(x, anchors):
    # x: (batch, d), anchors: (num_anchors, d)
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1).to(x.device)
    return torch.einsum("bm, am -> ba", x, anchors)

def relative_projection_euclidean(x, anchors):
    # Use negative Euclidean distances as similarity.
    # torch.cdist returns (batch, num_anchors)
    return - torch.cdist(x, anchors, p=2)

def relative_projection_mahalanobis_batched(x, anchors, inv_cov, batch_size=512):
    # Ensure anchors is a torch.Tensor
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)  # (B, num_anchors, d)
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        result.append(dists)
    return torch.cat(result, dim=0)

# def relative_projection_mahalanobis_batched_log(x, anchors, inv_cov, batch_size=512, epsilon=1e-8):
#     if not isinstance(anchors, torch.Tensor):
#         anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
#     result = []
#     for i in range(0, x.size(0), batch_size):
#         x_batch = x[i:i+batch_size]
#         diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)
#         dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + epsilon)
#         log_dists = torch.log1p(dists + epsilon)
#         result.append(log_dists)
#     return torch.cat(result, dim=0)

def relative_projection_mahalanobis_batched_log(x, anchors, inv_cov, batch_size=512, epsilon=1e-8, scale=5):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + epsilon)
        # Scale the distances inside the log
        effective_dists = scale * torch.log1p(dists / scale + epsilon)
        result.append(effective_dists)
    return torch.cat(result, dim=0)

def relative_projection_mahalanobis_batched_tanh(x, anchors, inv_cov, batch_size=512, scale=6.0):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        effective_dists = scale * torch.tanh(dists / scale)
        result.append(- effective_dists)
    return torch.cat(result, dim=0)

def relative_projection_mahalanobis_batched_exp(x, anchors, inv_cov, batch_size=512, scale=2.0):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        effective_dists = scale * (1 - torch.exp(- dists / scale))
        result.append(effective_dists)
    return torch.cat(result, dim=0)

def compute_covariance_matrix(features):
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    cov = (centered.t() @ centered) / (features.size(0) - 1)
    return cov


def evaluate_classifier(classifier, feats, labels, device):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    return 100 * f1_score(true_labels, preds, average="macro")

# ----- Dataset functions -----
def get_dataset(split, perc=1.0):
    dataset = load_dataset("cifar100", split=split)
    if perc < 1.0:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: int(len(dataset) * perc)]
        dataset = dataset.select(indices)
    return dataset

def collate_fn(batch, transform):
    images = [transform(sample["img"]) for sample in batch]
    labels = [sample["coarse_label"] for sample in batch]
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

def extract_features(model, dataloader, device):
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            feats = model(images)
            features_list.append(feats.cpu())
            labels_list.append(labels)
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels

def get_kmeans_anchors_clustered(src_emb, tgt_emb, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
    """
    Computes "parallel" anchors from two embeddings (source and target) using the same candidate indices.
    
    Steps:
      1. Run KMeans on the source embeddings (src_emb) with n_clusters=anchor_num.
      2. For each cluster center, compute Euclidean distances (in src_emb) and select the n_closest points.
      3. Compute optimized weights (with an entropy-like regularization) so that the weighted combination
         of these candidate points approximates the cluster center.
      4. Use the same candidate global indices (assuming same order in src_emb and tgt_emb) to compute
         weighted averages in both src_emb and tgt_emb.
    
    Args:
      src_emb: numpy array of shape [N, D_src] – used for clustering.
      tgt_emb: numpy array of shape [N, D_tgt] – parallel datapoints.
      anchor_num: number of anchors/clusters.
      n_closest: number of candidate points per cluster.
      kmeans_seed: random seed for KMeans.
      verbose: if True, prints debugging information.
      
    Returns:
      anchors_src: numpy array of shape [anchor_num, D_src] (approximated anchors from src_emb)
      anchors_tgt: numpy array of shape [anchor_num, D_tgt] (parallel anchors from tgt_emb)
      clusters_info: list of tuples for each cluster: (candidate_indices, weights, center)
                     where candidate_indices are indices in src_emb (and tgt_emb) used for the weighted average.
    """
    N = src_emb.shape[0]
    global_idx = np.arange(N)
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(src_emb)
    centers = kmeans.cluster_centers_
    
    clusters_info = []
    anchors_src = []
    anchors_tgt = []
    
    for center in tqdm(centers, desc="Processing clusters"):
        dists = np.linalg.norm(src_emb - center, axis=1)
        candidate_order = np.argsort(dists)[:n_closest]
        candidate_global = global_idx[candidate_order]  # These are global indices.
        candidate_points = src_emb[candidate_order]
        weights = optimize_weights(center, candidate_points)
        clusters_info.append((candidate_global, weights, center))
        # Use the same candidate indices to get weighted averages in both embeddings.
        anchor_src = np.average(src_emb[candidate_global], axis=0, weights=weights)
        anchor_tgt = np.average(tgt_emb[candidate_global], axis=0, weights=weights)
        anchors_src.append(anchor_src)
        anchors_tgt.append(anchor_tgt)
    
    anchors_src = np.vstack(anchors_src)
    anchors_tgt = np.vstack(anchors_tgt)
    
    if verbose:
        print("KMeans centers (first):", centers[0])
        print("First cluster candidates indices:", clusters_info[0][0])
        print("First approximated anchor (src):", anchors_src[0])
        print("First approximated anchor (tgt):", anchors_tgt[0])
    
    return anchors_src, anchors_tgt, clusters_info

def get_kmeans_anchors_unclustered(src_emb, tgt_emb, anchor_num, n_closest=2, kmeans_seed=42, verbose=False):
    """
    Computes "parallel" anchors from two embeddings (source and target) by approximating each KMeans center
    using candidate datapoints that are among the n_closest to the 20 closest centers (based on Euclidean distance
    between centers).

    For each center (of anchor_num clusters):
        1. Find the 20 closest centers (excluding the current center).
        2. For each of these centers, select the n_closest datapoints (using Euclidean distance).
        3. Use the union of those candidate indices for weight optimization (fallback to global indices if too few).
        4. Compute optimized weights and then the weighted anchor (in both source and target spaces).
    """
    N = src_emb.shape[0]
    global_idx = np.arange(N)
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(src_emb)
    centers = kmeans.cluster_centers_

    # Precompute the n_closest points for each center (for each center individually)
    candidate_lists = []
    for center in centers:
        dists = np.linalg.norm(src_emb - center, axis=1)
        candidate_order = np.argsort(dists)[:n_closest]
        candidate_lists.append(candidate_order)

    clusters_info = []
    anchors_src = []
    anchors_tgt = []

    closest_center_num = 20
    for i, center in enumerate(tqdm(centers, desc="Processing clusters")):
        # Compute distances between the current center and all other centers.
        center_dists = np.linalg.norm(centers - center, axis=1)
        center_dists[i] = np.inf  # Exclude self.
        # Select indices of the 20 closest centers.
        closest_indices = np.argsort(center_dists)[:closest_center_num]
        # Combine candidate indices from these 20 closest centers.
        other_candidates = np.concatenate([candidate_lists[j] for j in closest_indices])
        candidate_idxs = np.unique(other_candidates)
        # Fallback if the candidate set is too small.
        if candidate_idxs.size < n_closest:
            print("Removed too many duplicates. Falling back to all global indices.")
            candidate_idxs = global_idx
        candidate_points = src_emb[candidate_idxs]
        weights = optimize_weights(center, candidate_points, lambda_reg=0.40)
        clusters_info.append((candidate_idxs, weights, center))
        anchor_src = np.average(src_emb[candidate_idxs], axis=0, weights=weights)
        anchor_tgt = np.average(tgt_emb[candidate_idxs], axis=0, weights=weights)
        anchors_src.append(anchor_src)
        anchors_tgt.append(anchor_tgt)

    anchors_src = np.vstack(anchors_src)
    anchors_tgt = np.vstack(anchors_tgt)

    if verbose:
        print("KMeans centers (first):", centers[0])
        print("First cluster candidate indices:", clusters_info[0][0])
        print("First approximated anchor (src):", anchors_src[0])
        print("First approximated anchor (tgt):", anchors_tgt[0])

    return anchors_src, anchors_tgt, clusters_info

def build_classifier(input_dim, intermediate_dim, num_classes, dropout_p=0.5):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, intermediate_dim),
        nn.Tanh(),
        nn.Dropout(p=dropout_p),  # added dropout
        Lambda(lambda x: x.permute(1, 0)),
        nn.InstanceNorm1d(intermediate_dim),
        Lambda(lambda x: x.permute(1, 0)),
        nn.Linear(intermediate_dim, num_classes)
    )

def train_classifier(classifier, train_feats, train_labels, device, num_epochs=7, weight_decay=1e-4):
    classifier.train()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=weight_decay)  # weight_decay added
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.size(0)
        avg_loss = running_loss / len(dataset)
        print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return classifier

# ----- Main experiment -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_perc = 1.0
    batch_size = 64
    num_epochs = 9
    n_closest = 100 # number of closest points to each KMeans or P center
    n_seeds = 7  # seeds per anchor configuration

    anchor_nums = list(range(100, 601, 50))

    transformer_names = [ #to copy and paste from
    "vit_base_patch16_224",
    "rexnet_100",
    "vit_base_resnet50_384",
    "vit_small_patch16_224"
    ]

    decoder_transformer = "rexnet_100"
    zero_shot_transformer = "vit_small_patch16_224"

    print("Loading CIFAR-100 dataset...")
    train_dataset = get_dataset("train", perc=train_perc)
    test_dataset = get_dataset("test", perc=train_perc)
    if hasattr(train_dataset.features["coarse_label"], "num_classes"):
        num_classes = train_dataset.features["coarse_label"].num_classes
    else:
        num_classes = 20

    # --- Load or extract features ---
    current_path = os.path.dirname(os.path.abspath(__file__))
    features_file = os.path.join(current_path, "features_dict_CIFAR100_coarse.pt")
    if os.path.exists(features_file):
        features_dict = torch.load(features_file)
        print(f"Loaded precomputed features from {features_file}")
    else:
        features_dict = {}
        for model_name in [decoder_transformer, zero_shot_transformer]:
            print(f"\nExtracting features for model: {model_name}")
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            model.to(device)
            model.eval()
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: collate_fn(batch, transform))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: collate_fn(batch, transform))
            train_feats, train_labels = extract_features(model, train_loader, device)
            test_feats, test_labels = extract_features(model, test_loader, device)
            features_dict[model_name] = {
                "train_features": train_feats,
                "train_labels": train_labels,
                "test_features": test_feats,
                "test_labels": test_labels,
            }
        torch.save(features_dict, features_file)
        print(f"Saved precomputed features to {features_file}")
    
    # For training the relative decoder, use features from decoder_transformer.
    enc1_feats_train = features_dict[decoder_transformer]["train_features"]
    enc1_labels_train = features_dict[decoder_transformer]["train_labels"]
    enc1_feats_test = features_dict[decoder_transformer]["test_features"]
    enc1_labels_test = features_dict[decoder_transformer]["test_labels"]
    
    # For evaluation (zero-shot), use features from zero_shot_transformer.
    enc2_feats_train = features_dict[zero_shot_transformer]["train_features"]
    enc2_feats_test = features_dict[zero_shot_transformer]["test_features"]
    enc2_labels_test = features_dict[zero_shot_transformer]["test_labels"]

    emb_list = [enc1_feats_test.cpu().numpy(), enc2_feats_test.cpu().numpy()] # For anchors
    
    # --- Experiment: Loop over anchor numbers ---
    results_random = {a: [] for a in anchor_nums}
    results_optimized = {a: [] for a in anchor_nums}
    results_kmeans = {a: [] for a in anchor_nums}
    
    # For each configuration, we use the same random anchor indices (fixed across models)
    for num_anchors in anchor_nums:
        print(f"\nEvaluating for {num_anchors} anchors...")
        for seed in range(42, 42 + n_seeds):
            print(f" Seed {seed}")
            set_random_seeds(seed)
            sample_count = enc1_feats_train.shape[0]
            # FIXED RANDOM ANCHORS: compute once and reuse for both models
            random_indices = np.sort(np.random.choice(sample_count, num_anchors, replace=False))
            
            # -------- Training phase --------
            # Random method: use the same fixed indices for training features
            anchors_random_train = enc1_feats_train[random_indices].to(device)
            rel_train_random = relative_projection_cosine(enc1_feats_train.to(device), anchors_random_train)
            
            # Optimized method: optimize anchors on training features (decoder_transformer)
            coverage_w = 0.95
            diversity_w = 1 - coverage_w
            anti_collapse_w = 0
            clusterd_P_anchors = True
            
            # if clusterd_P_anchors:
            #     _, P_anchors, _ = get_P_anchors_clustered(
            #         emb=emb_list,
            #         anchor_num=num_anchors,
            #         n_closest=n_closest,
            #         epochs=200,
            #         lr=1e-2,
            #         coverage_weight=coverage_w,
            #         diversity_weight=diversity_w,
            #         anti_collapse_w=anti_collapse_w,
            #         exponent=1,
            #         dist_measure="euclidean",
            #         verbose=True,
            #         device=device
            #     )
            # else:
            #     _, P_anchors = get_optimized_anchors(
            #         emb=emb_list,
            #         anchor_num=num_anchors,
            #         epochs=200,
            #         lr=1e-2,
            #         coverage_weight=coverage_w,
            #         diversity_weight=diversity_w,
            #         anti_collapse_w=anti_collapse_w,
            #         exponent=1,
            #         dist_measure="euclidean",
            #         verbose=False,
            #         device=device
            #     )
            
            # P_anchors_enc1, P_anchors_enc2 = P_anchors[0], P_anchors[1]

            # print("Computing cov matrix")
            # cov_train = compute_covariance_matrix(enc1_feats_train.to(device))
            # inv_cov_train = torch.linalg.inv(cov_train + 1e-6 * torch.eye(cov_train.size(0), device=cov_train.device))
            # print("Computing relative projection")
            # rel_train_optimized = relative_projection_mahalanobis_batched(enc1_feats_train.to(device),
            #                                                              P_anchors_enc1, inv_cov_train)

            # # Train classifiers on the relative representations
            # clf_random = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            # clf_optimized = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            # print(" Training Random relative decoder:")
            # clf_random = train_classifier(clf_random, rel_train_random, enc1_labels_train.to(device),
            #                               device, num_epochs=num_epochs)
            # print(" Training Optimized relative decoder:")
            # clf_optimized = train_classifier(clf_optimized, rel_train_optimized, enc1_labels_train.to(device),
            #                                  device, num_epochs=num_epochs)
            
            # # -------- Testing phase (zero_shot_transformer features) --------
            # # Using the SAME fixed random indices on zero_shot_transformer training features
            # anchors_random_test = enc2_feats_train[random_indices].to(device)
            # rel_test_random = relative_projection_cosine(enc2_feats_test.to(device), anchors_random_test)
            # f1_random = evaluate_classifier(clf_random, rel_test_random, enc2_labels_test.to(device), device)
            
            # cov_test = compute_covariance_matrix(enc2_feats_train.to(device))
            # inv_cov_test = torch.linalg.inv(cov_test + 1e-6 * torch.eye(cov_test.size(0), device=cov_test.device))
            # rel_test_optimized = relative_projection_mahalanobis_batched(enc2_feats_test.to(device),
            #                                                             P_anchors_enc2, inv_cov_test)
            # f1_optimized = evaluate_classifier(clf_optimized, rel_test_optimized, enc2_labels_test.to(device), device)
            
            # print(f"  Random (cosine) F1: {f1_random:.2f}%, Optimized (mahalanobis) F1: {f1_optimized:.2f}%")
            # results_random[num_anchors].append(f1_random)
            # results_optimized[num_anchors].append(f1_optimized)

            # ### TESTING KMEANS ANCHORS ###
            print("Computing KMeans-based datapoint anchors on enc1")
            anchors_enc1_np, anchors_enc2_np, _ = get_kmeans_anchors_clustered(
                src_emb=enc1_feats_test.cpu().numpy(),
                tgt_emb=enc2_feats_test.cpu().numpy(),
                anchor_num=num_anchors,
                n_closest=n_closest,
                kmeans_seed=seed,
                verbose=False
            )

            # Convert to torch tensors as needed:
            anchors_enc1 = torch.tensor(anchors_enc1_np, device=device, dtype=enc1_feats_test.dtype)
            anchors_enc2 = torch.tensor(anchors_enc2_np, device=device, dtype=enc2_feats_test.dtype)

            # ------------------------- Training -----------------------------
            # Compute covariance/inverse cov on enc1 training features and compute relative representations
            cov_enc1 = compute_covariance_matrix(enc1_feats_train.to(device))
            inv_cov_enc1 = torch.linalg.inv(cov_enc1 + 1e-6 * torch.eye(cov_enc1.size(0), device=cov_enc1.device))
            rel_train_kmeans = relative_projection_mahalanobis_batched(
                enc1_feats_train.to(device),
                anchors_enc1,
                inv_cov_enc1
            )

            # Train a classifier using the enc1-based relative representations (with enc1 labels)
            clf_kmeans = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            print(" Training KMeans relative decoder (train on enc1 relrep):")
            clf_kmeans = train_classifier(clf_kmeans, rel_train_kmeans, enc1_labels_train.to(device),
                                        device, num_epochs=num_epochs)
            
            ### TESTING NO ZERO SHOT CLASSIFIER PERFORMANCE TO CHECK FOR OVERFITTING ###
            # cov_enc1_test = compute_covariance_matrix(enc1_feats_test.to(device))
            # inv_cov_enc1_test = torch.linalg.inv(cov_enc1_test + 1e-6 * torch.eye(cov_enc1_test.size(0), device=cov_enc1_test.device))
            # rel_test_kmeans = relative_projection_mahalanobis_batched(
            #     enc1_feats_test.to(device),
            #     anchors_enc1,
            #     inv_cov_enc1_test
            # )

            # f1_kmeans = evaluate_classifier(clf_kmeans, rel_test_kmeans, enc1_labels_test.to(device), device)
            # print(f"  KMeans (mahalanobis) NO ZERO SHOT test data F1: {f1_kmeans:.2f}%")
            # f1_kmeans = evaluate_classifier(clf_kmeans, rel_train_kmeans, enc1_labels_train.to(device), device)
            # print(f"  KMeans (mahalanobis) NO ZERO SHOT train data F1: {f1_kmeans:.2f}%")


            # ------------------------- Testing -----------------------------
            # For testing, select the corresponding enc2 anchors using the same medoid indices.
            cov_enc2 = compute_covariance_matrix(enc2_feats_test.to(device))
            inv_cov_enc2 = torch.linalg.inv(cov_enc2 + 1e-6 * torch.eye(cov_enc2.size(0), device=cov_enc2.device))
            rel_test_kmeans = relative_projection_mahalanobis_batched(
                enc2_feats_test.to(device),
                anchors_enc2,
                inv_cov_enc2
            )
            f1_kmeans = evaluate_classifier(clf_kmeans, rel_test_kmeans, enc2_labels_test.to(device), device)
            print(f"  KMeans (mahalanobis) F1: {f1_kmeans:.2f}%")
            results_kmeans[num_anchors].append(f1_kmeans)


            ### TESTING KMEANS WITH EUCLIDEAN AND COSINE SIMILARITY ###
            
            # # ----- Euclidean Relative Representations Kmeans -----
            # print("Computing Euclidean relative projections using KMeans anchors")
            # # Compute relative representations on training features using negative euclidean distances.
            # rel_train_kmeans_eucl = relative_projection_euclidean(enc1_feats_train.to(device), anchors_enc1)
            # # Retrain a new classifier using these Euclidean relative representations.
            # clf_kmeans_eucl = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            # print(" Training KMeans relative decoder (Euclidean) on enc1 relrep:")
            # clf_kmeans_eucl = train_classifier(clf_kmeans_eucl, rel_train_kmeans_eucl, 
            #                                 enc1_labels_train.to(device), device, num_epochs=8)
            # # Compute test relative representations for Euclidean projection on target features.
            # rel_test_kmeans_eucl = relative_projection_euclidean(enc2_feats_test.to(device), anchors_enc2)
            # f1_kmeans_eucl = evaluate_classifier(clf_kmeans_eucl, rel_test_kmeans_eucl, 
            #                                     enc2_labels_test.to(device), device)
            # print(f"  KMeans (euclidean) F1: {f1_kmeans_eucl:.2f}%")

            # # ----- Cosine Relative Representations Kmeans -----
            # print("Computing Cosine relative projections using KMeans anchors")
            # # Compute relative representations on training features using cosine similarity.
            # rel_train_kmeans_cos = relative_projection_cosine(enc1_feats_train.to(device), anchors_enc1)
            # # Retrain a new classifier for cosine-based projections.
            # clf_kmeans_cos = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            # print(" Training KMeans relative decoder (cosine) on enc1 relrep:")
            # clf_kmeans_cos = train_classifier(clf_kmeans_cos, rel_train_kmeans_cos, 
            #                                 enc1_labels_train.to(device), device, num_epochs=8)
            # # Compute test relative representations for cosine projection on target features.
            # rel_test_kmeans_cos = relative_projection_cosine(enc2_feats_test.to(device), anchors_enc2)
            # f1_kmeans_cos = evaluate_classifier(clf_kmeans_cos, rel_test_kmeans_cos, 
            #                                     enc2_labels_test.to(device), device)
            # print(f"  KMeans (cosine) F1: {f1_kmeans_cos:.2f}%")
            
    
    # --- Compute mean and std for each anchor number ---
    anchor_nums_arr = []
    random_means = []
    random_stds = []
    optimized_means = []
    optimized_stds = []
    kmeans_means = []
    kmeans_stds = []
    for a in anchor_nums:
        anchor_nums_arr.append(a)
        r_mean = np.mean(results_random[a])
        r_std = np.std(results_random[a])
        o_mean = np.mean(results_optimized[a])
        o_std = np.std(results_optimized[a])
        k_mean = np.mean(results_kmeans[a])
        k_std = np.std(results_kmeans[a])
        random_means.append(r_mean)
        random_stds.append(r_std)
        optimized_means.append(o_mean)
        optimized_stds.append(o_std)
        kmeans_means.append(k_mean)
        kmeans_stds.append(k_std)
        print(f"Anchors: {a}, Random: {r_mean:.2f}% ± {r_std:.2f}%, Optimized: {o_mean:.2f}% ± {o_std:.2f}%, KMeans: {k_mean:.2f}% ± {k_std:.2f}%")

    # --- Save results to CSV ---
    csv_file = os.path.join(current_path, f"gred_vs_rand_dec_{decoder_transformer}_enc_{zero_shot_transformer}.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Anchors", "Random_mean", "Random_std", "Optimized_mean", "Optimized_std", "KMeans_mean", "KMeans_std"])
        for a, r_mean, r_std, o_mean, o_std, k_mean, k_std in zip(anchor_nums_arr, random_means, random_stds, optimized_means, optimized_stds, kmeans_means, kmeans_stds):
            writer.writerow([a, r_mean, r_std, o_mean, o_std, k_mean, k_std])
    print(f"Results saved to {csv_file}")

    # --- Plot the results ---
    plt.figure()
    plt.plot(anchor_nums_arr, random_means, label="Random (cosine)", marker='o')
    plt.fill_between(anchor_nums_arr,
                    np.array(random_means) - np.array(random_stds),
                    np.array(random_means) + np.array(random_stds),
                    alpha=0.2)
    plt.plot(anchor_nums_arr, optimized_means, label="Optimized (mahalanobis)", marker='o')
    plt.fill_between(anchor_nums_arr,
                    np.array(optimized_means) - np.array(optimized_stds),
                    np.array(optimized_means) + np.array(optimized_stds),
                    alpha=0.2)
    plt.plot(anchor_nums_arr, kmeans_means, label="KMeans (mahalanobis)", marker='o')
    plt.fill_between(anchor_nums_arr,
                    np.array(kmeans_means) - np.array(kmeans_stds),
                    np.array(kmeans_means) + np.array(kmeans_stds),
                    alpha=0.2)
    plt.xlabel("Number of Anchors")
    plt.ylabel("F1 Score (%)")
    plt.title("Relative Decoder Performance vs Number of Anchors")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
