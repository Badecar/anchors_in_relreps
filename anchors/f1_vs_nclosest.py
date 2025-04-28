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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
import csv
from tqdm import tqdm

os.environ["LOKY_MAX_CPU_COUNT"] = "16"
USE_TQDM = False

# Import the weight optimization routine from your project.
from P_anchors import optimize_weights
from utils import set_random_seeds

# Lambda module for wrapping lambda functions in nn.Sequential
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

# ----- Relative projection function using Mahalanobis (batched) -----
def compute_covariance_matrix(features):
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    cov = (centered.t() @ centered) / (features.size(0) - 1)
    return cov

def relative_projection_mahalanobis_batched(x, anchors, inv_cov, batch_size=512):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        result.append(-dists)
    return torch.cat(result, dim=0)

# ----- KMeans-based anchor extraction with weight optimization -----
def get_kmeans_anchors_clustered(src_emb, tgt_emb, anchor_num, n_closest, kmeans_seed=42, verbose=False):
    N = src_emb.shape[0]
    global_idx = np.arange(N)
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(src_emb)
    centers = kmeans.cluster_centers_

    clusters_info = []
    anchors_src = []
    anchors_tgt = []

    iterator = tqdm(centers, desc=f"Clusters (n_closest={n_closest})", leave=False) if USE_TQDM else centers
    for center in iterator:
        dists = np.linalg.norm(src_emb - center, axis=1)
        candidate_order = np.argsort(dists)[:n_closest]
        candidate_global = global_idx[candidate_order]
        candidate_points = src_emb[candidate_order]
        weights = optimize_weights(center, candidate_points)
        clusters_info.append((candidate_global, weights, center))
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

# ----- Classifier building, training and evaluation -----
def build_classifier(input_dim, intermediate_dim, num_classes, dropout_p=0.5):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, intermediate_dim),
        nn.Tanh(),
        nn.Dropout(p=dropout_p),
        Lambda(lambda x: x.permute(1, 0)),
        nn.InstanceNorm1d(intermediate_dim),
        Lambda(lambda x: x.permute(1, 0)),
        nn.Linear(intermediate_dim, num_classes)
    )

def train_classifier(classifier, train_feats, train_labels, device, num_epochs=7, weight_decay=1e-4):
    classifier.train()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=weight_decay)
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
    return classifier.eval()

def evaluate_classifier(classifier, feats, labels, device):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    return 100 * f1_score(true_labels, preds, average="macro")

# ----- Main experiment: varying n_closest for KMeans anchors with Mahalanobis -----
if __name__ == "__main__":
    anchor_num = 200  # fixed number of anchors
    # n_closest values: starting with 1 then 5,10,...,100
    n_closest_list = [1] + list(range(5, 101, 5))
    seeds = list(range(42, 42 + 5))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load precomputed features.
    features_file = os.path.join(current_path, "features_dict_CIFAR100_coarse.pt")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Please run the feature extraction script first.")
    features_dict = torch.load(features_file)
    
    # Use only the selected base and target encoders.
    base_encoder = "vit_base_resnet50_384"
    target_encoder = "vit_small_patch16_224"
    if base_encoder not in features_dict or target_encoder not in features_dict:
        raise KeyError("Selected base or target encoder not found in the features file.")

    base_train_feats = features_dict[base_encoder]["train_features"]
    base_train_labels = features_dict[base_encoder]["train_labels"]
    base_test_feats = features_dict[base_encoder]["test_features"]

    target_test_feats = features_dict[target_encoder]["test_features"]
    target_test_labels = features_dict[target_encoder]["test_labels"]
    sample_key = base_encoder
    num_classes = int(features_dict[sample_key]["train_labels"].max().item() + 1)

    # Results will be stored in a dict: {n_closest: [F1 scores]}
    results = {n_closest: [] for n_closest in n_closest_list}

    print(f"\n========== Running experiments with base encoder: {base_encoder} and target encoder: {target_encoder} ==========")
    for n_closest in n_closest_list:
        print(f"\nEvaluating with n_closest: {n_closest} ...")
        for seed in seeds:
            set_random_seeds(seed)
            # ---- SELECT ANCHORS using KMeans (only method) ----
            anchors_base_np, anchors_target_np, _ = get_kmeans_anchors_clustered(
                src_emb=base_test_feats.cpu().numpy(),
                tgt_emb=target_test_feats.cpu().numpy(),
                anchor_num=anchor_num,
                n_closest=n_closest,
                kmeans_seed=seed,
                verbose=False
            )
            anchors_base = torch.tensor(anchors_base_np, device=device, dtype=base_test_feats.dtype)
            anchors_target = torch.tensor(anchors_target_np, device=device, dtype=target_test_feats.dtype)

            # ---- TRAIN PHASE ----
            cov_base = compute_covariance_matrix(base_train_feats.to(device))
            inv_cov_base = torch.linalg.inv(
                cov_base + 1e-6 * torch.eye(cov_base.size(0), device=cov_base.device)
            )
            rel_train = relative_projection_mahalanobis_batched(
                base_train_feats.to(device), anchors_base, inv_cov_base
            )
            clf = build_classifier(anchor_num, anchor_num, num_classes).to(device)
            clf = train_classifier(clf, rel_train, base_train_labels.to(device), device, num_epochs=7)

            # ---- TEST PHASE ----
            cov_target = compute_covariance_matrix(target_test_feats.to(device))
            inv_cov_target = torch.linalg.inv(
                cov_target + 1e-6 * torch.eye(cov_target.size(0), device=cov_target.device)
            )
            rel_test = relative_projection_mahalanobis_batched(
                target_test_feats.to(device), anchors_target, inv_cov_target
            )
            f1 = evaluate_classifier(clf, rel_test, target_test_labels.to(device), device)
            results[n_closest].append(f1)
            print(f" Base: {base_encoder}, Target: {target_encoder}, n_closest: {n_closest}, Seed: {seed}, F1: {f1:.2f}%")

    # Write results to CSV.
    csv_file = os.path.join(current_path, f"results_nclosestplot_{anchor_num}anch_kmeans_mahalanobis_base_{base_encoder}_target_{target_encoder}.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n_closest", "Mean_F1", "Std_F1"])
        for n_closest in n_closest_list:
            scores = np.array(results[n_closest])
            writer.writerow([n_closest, scores.mean(), scores.std()])
    print(f"\nSaved results to {csv_file}")