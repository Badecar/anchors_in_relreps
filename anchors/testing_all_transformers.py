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
USE_TQDM = True

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

def relative_projection_cosine(x, anchors):
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1).to(x.device)
    return torch.einsum("bm, am -> ba", x, anchors)

# ----- KMeans-based anchor extraction with weight optimization ----- 
def get_kmeans_anchors_clustered(src_emb, tgt_emb, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
    N = src_emb.shape[0]
    global_idx = np.arange(N)
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(src_emb)
    centers = kmeans.cluster_centers_
    
    clusters_info = []
    anchors_src = []
    anchors_tgt = []
    
    iterator = tqdm(centers, desc="Processing clusters", leave=False) if USE_TQDM else centers
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
        nn.Dropout(p=dropout_p),  # added dropout
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

# ----- Main experiment: using precomputed KMeans centers from base transformer -----
if __name__ == "__main__":
    # Set the methods directly in the code.
    anchor_method = "random"      # Options: "kmeans" or "random"
    dist_metric   = "cosine" # Options: "mahalanobis" or "cosine"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load precomputed features.
    current_path = os.path.dirname(os.path.abspath(__file__))
    features_file = os.path.join(current_path, "features_dict_CIFAR100_coarse.pt")
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"{features_file} not found. Please run the feature extraction script first.")
    features_dict = torch.load(features_file)
    transformers = list(features_dict.keys())
    
    sample_key = transformers[0]
    num_classes = int(features_dict[sample_key]["train_labels"].max().item() + 1)
    
    # Experiment parameters.
    anchor_nums = [768]  # Use only 768 anchors.
    seeds = list(range(42, 42 + 20))
    n_closest = 100 
    
    for base in transformers:
        print(f"\n========== Running experiments with base transformer: {base} ==========")
        base_train_feats = features_dict[base]["train_features"]
        base_train_labels = features_dict[base]["train_labels"]
        base_test_feats = features_dict[base]["test_features"]
        base_test_labels = features_dict[base]["test_labels"]
        
        results = {a: {target: [] for target in transformers} for a in anchor_nums}
        
        for num_anchors in anchor_nums:
            print(f"\nEvaluating with {num_anchors} anchors...")
            for seed in seeds:
                set_random_seeds(seed)
                for target in transformers:
                    target_test_feats = features_dict[target]["test_features"]
                    target_test_labels = features_dict[target]["test_labels"]
                    
                    # ---- SELECT ANCHORS ----
                    if anchor_method == "kmeans":
                        anchors_base_np, anchors_target_np, _ = get_kmeans_anchors_clustered(
                            src_emb=base_test_feats.cpu().numpy(),
                            tgt_emb=target_test_feats.cpu().numpy(),
                            anchor_num=num_anchors,
                            n_closest=n_closest,
                            kmeans_seed=seed,
                            verbose=False
                        )
                        anchors_base = torch.tensor(anchors_base_np, device=device, dtype=base_test_feats.dtype)
                        anchors_target = torch.tensor(anchors_target_np, device=device, dtype=target_test_feats.dtype)
                    elif anchor_method == "random":
                        # randomly select anchor indices from test features.
                        random_indices = np.sort(
                            np.random.choice(base_test_feats.shape[0], num_anchors, replace=False)
                        )
                        anchors_base = base_test_feats[random_indices].to(device)
                        anchors_target = target_test_feats[random_indices].to(device)
                    
                    # ---- TRAIN PHASE ----
                    if dist_metric == "mahalanobis":
                        cov_base = compute_covariance_matrix(base_train_feats.to(device))
                        inv_cov_base = torch.linalg.inv(
                            cov_base + 1e-6 * torch.eye(cov_base.size(0), device=cov_base.device)
                        )
                        rel_train = relative_projection_mahalanobis_batched(
                            base_train_feats.to(device), anchors_base, inv_cov_base
                        )
                    elif dist_metric == "cosine":
                        rel_train = relative_projection_cosine(base_train_feats.to(device), anchors_base)
                    
                    clf = build_classifier(num_anchors, num_anchors, num_classes).to(device)
                    clf = train_classifier(clf, rel_train, base_train_labels.to(device), device, num_epochs=7)
                    
                    # ---- TEST PHASE ----
                    if dist_metric == "mahalanobis":
                        cov_target = compute_covariance_matrix(target_test_feats.to(device))
                        inv_cov_target = torch.linalg.inv(
                            cov_target + 1e-6 * torch.eye(cov_target.size(0), device=cov_target.device)
                        )
                        rel_test = relative_projection_mahalanobis_batched(
                            target_test_feats.to(device), anchors_target, inv_cov_target
                        )
                    elif dist_metric == "cosine":
                        rel_test = relative_projection_cosine(target_test_feats.to(device), anchors_target)
                    
                    f1 = evaluate_classifier(clf, rel_test, target_test_labels.to(device), device)
                    results[num_anchors][target].append(f1)
                    print(f" Base: {base}, Target: {target}, Anchors: {num_anchors}, Seed: {seed}, F1: {f1:.2f}%")
        
        # Write results to CSV.
        csv_file = os.path.join(current_path, f"results_{anchor_method}_{dist_metric}_base_{base}.csv")
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Anchors", "Target", "Mean_F1", "Std_F1"])
            for a in anchor_nums:
                for target in results[a]:
                    scores = np.array(results[a][target])
                    writer.writerow([a, target, scores.mean(), scores.std()])
        print(f"\nSaved results for base transformer '{base}' to {csv_file}")