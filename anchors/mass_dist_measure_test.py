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
from utils import set_random_seeds
from P_anchors import get_optimized_anchors, AnchorSelector
original_forward = AnchorSelector.forward
def patched_forward(self, X):
    result = original_forward(self, X)
    if isinstance(result, tuple):
        return result[0]
    return result
AnchorSelector.forward = patched_forward

# A simple Lambda module to wrap lambda functions inside nn.Sequential
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

# ----- NEW: Relative projection functions for different metrics -----
def relative_projection_cosine(x, anchors):
    # x: (batch, d), anchors: (num_anchors, d)
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1).to(x.device)
    return torch.einsum("bm, am -> ba", x, anchors)

def relative_projection_euclidean(x, anchors):
    # Use negative Euclidean distances as similarity.
    # torch.cdist returns (batch, num_anchors)
    return - torch.cdist(x, anchors, p=2)

def compute_covariance_matrix(features):
    # features: (n, d)
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    cov = (centered.t() @ centered) / (features.size(0)-1)
    return cov

def relative_projection_mahalanobis(x, anchors, inv_cov):
    # x: (batch, d), anchors: (num_anchors, d), inv_cov: (d,d)
    diff = x.unsqueeze(1) - anchors.unsqueeze(0)  # (batch, num_anchors, d)
    distances = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff)+1e-8)
    return - distances


# Batched Mahalanobis projection to avoid OOM
def relative_projection_mahalanobis_batched(x, anchors, inv_cov, batch_size=1024):
    # x: (N, d), anchors: (num_anchors, d)
    # We'll compute distances for x in batches.
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]              # shape: (B, d)
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)  # (B, num_anchors, d)
        # Use the same einsum string as in the non-batched version:
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8) #NOTE: not quite sure about the string here. Maybe "bid,ij,bid->bi" ?
        result.append(-dists)
    return torch.cat(result, dim=0)

# --------------------------------------------------------------------

# Get the dataset (with an optional percentage of samples)
def get_dataset(split: str, perc: float = 1.0):
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

def build_classifier(input_dim, intermediate_dim, num_classes):
    return nn.Sequential(
        nn.LayerNorm(normalized_shape=input_dim),
        nn.Linear(in_features=input_dim, out_features=intermediate_dim),
        nn.Tanh(),
        Lambda(lambda x: x.permute(1, 0)),
        nn.InstanceNorm1d(num_features=intermediate_dim),
        Lambda(lambda x: x.permute(1, 0)),
        nn.Linear(in_features=intermediate_dim, out_features=num_classes)
    )

def train_classifier(classifier, train_feats, train_labels, test_feats, test_labels, device, num_epochs=5, description=""):
    classifier.train()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * feats.size(0)
        avg_loss = running_loss / len(train_dataset)
        f1 = evaluate_classifier(classifier, test_feats, test_labels, device)
        print(f"[{description}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test F1: {f1:.2f}%")

def evaluate_classifier(classifier, feats, labels, device):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    f1 = f1_score(true_labels, preds, average="macro")
    return 100 * f1

# Lists of transformer model names.
transformer_names = [
    "vit_base_patch16_224",
    "rexnet_100",
    "vit_base_resnet50_384", # Takes a long time to run
    "vit_small_patch16_224"
]
decoder_transformer_names = [
    "vit_base_patch16_224",
    "rexnet_100",
    "vit_base_resnet50_384",
    "vit_small_patch16_224"
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----- Fixed parameters -----
    train_perc = 1.0
    fine_grained = False
    target_key = "coarse_label" if not fine_grained else "fine_label"
    num_anchors = 768
    num_epochs = 7
    batch_size = 64
    coverage_w = 0.9
    diversity_w = 1 - coverage_w
    anti_collapse_w = 0
    anchor_encoder = 'vit_base_patch16_224'
    P_dist_measure = "euclidean"  # "euclidean", "cosine", or "mahalanobis"
    n_seeds = 4

    print("Loading the CIFAR-100 dataset...")
    train_dataset = get_dataset("train", perc=train_perc)
    test_dataset = get_dataset("test", perc=train_perc)
    if hasattr(train_dataset.features[target_key], "num_classes"):
        num_classes = train_dataset.features[target_key].num_classes
    else:
        num_classes = 20 if not fine_grained else 100

    #####################################
    # Precompute (or load) features for all models
    #####################################
    all_model_names = set(decoder_transformer_names + transformer_names)
    print("Models:", all_model_names)
    features_file = os.path.join(current_path, "features_dict_CIFAR100_coarse.pt")
    if os.path.exists(features_file):
        features_dict = torch.load(features_file)
        print(f"Loaded precomputed features from {features_file}")
    else:
        features_dict = dict()
        for model_name in all_model_names:
            print(f"\n=== Extracting features for model: {model_name} ===")
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            model.to(device)
            model.eval()
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, transform)
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, transform)
            )
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

    # Fix random anchor indices (same for all models)
    sample_count = features_dict[list(features_dict.keys())[0]]["train_features"].shape[0]
    random_anchor_indices = np.sort(np.random.choice(sample_count, num_anchors, replace=False))
    
    #####################################
    # Run experiments over n_seeds
    #####################################
    # We'll record results in dictionaries keyed by baseline model and then by metric and method.
    # For each baseline (decoder), we record:
    #   absolute  : F1 obtained using absolute classifier.
    #   optimized : {metric: {transformer: F1}}
    #   random    : {metric: {transformer: F1}}
    results = {baseline: {"absolute": [], "optimized": {"cosine":{}, "euclidean":{}, "mahalanobis":{}},
                          "random": {"cosine":{}, "euclidean":{}, "mahalanobis":{}}}
               for baseline in decoder_transformer_names}

    for seed in range(42, 42+n_seeds):
        print(f"\n===== Seed {seed} =====")
        set_random_seeds(seed)
        
        # Train anchors using the anchor encoder features.
        if anchor_encoder in features_dict:
            anchor_model_name = anchor_encoder
        else:
            anchor_model_name = list(features_dict.keys())[0]
        anchor_train_feats = features_dict[anchor_model_name]["train_features"]
        anchor_selector, _ = get_optimized_anchors(
            emb=[anchor_train_feats.cpu().numpy()],
            anchor_num=num_anchors,
            epochs=200,
            lr=1e-2,
            coverage_weight=coverage_w,
            diversity_weight=diversity_w,
            anti_collapse_w=anti_collapse_w,
            exponent=1,
            dist_measure=P_dist_measure,
            verbose=True,
            device=device
        )
        for param in anchor_selector.parameters():
            param.requires_grad = False

        # Loop over each baseline (decoder)
        for baseline in decoder_transformer_names:
            base_feats = features_dict[baseline]
            train_features_base = base_feats["train_features"].to(device)
            train_labels_base = base_feats["train_labels"].to(device)
            test_features_base = base_feats["test_features"].to(device)
            test_labels_base = base_feats["test_labels"].to(device)

            # Train absolute classifier (once per baseline)
            abs_classifier = build_classifier(
                input_dim=train_features_base.shape[1],
                intermediate_dim=num_anchors,
                num_classes=num_classes
            ).to(device)
            print(f"\nTraining absolute classifier for baseline {baseline} ...")
            train_classifier(abs_classifier, train_features_base, train_labels_base,
                               test_features_base, test_labels_base, device,
                               num_epochs=num_epochs, description="Absolute")
            abs_f1 = evaluate_classifier(abs_classifier, test_features_base, test_labels_base, device)
            results[baseline]["absolute"].append(abs_f1)
            
            # For relative classifiers, we prepare optimized and random anchors.
            anch_optimized = anchor_selector(train_features_base.to(device))
            anch_random = train_features_base[random_anchor_indices]
            # For Mahalanobis, compute inverse covariance based on training features.
            cov = compute_covariance_matrix(train_features_base)
            inv_cov = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.size(0), device=cov.device))
            
            # For each metric, compute training relative representations and train a classifier.
            proj_funcs = {
                "cosine": relative_projection_cosine,
                "mahalanobis": lambda x, anchors: relative_projection_mahalanobis_batched(x, anchors, inv_cov),
                "euclidean": relative_projection_euclidean
            }
            classifiers_opt = {}
            classifiers_rand = {}
            for metric, func in proj_funcs.items():
                rel_train_opt = func(train_features_base, anch_optimized)
                rel_train_rand = func(train_features_base, anch_random)
                # Build classifier for relative representations (input_dim = num_anchors).
                clf_opt = build_classifier(
                    input_dim=num_anchors,
                    intermediate_dim=num_anchors,
                    num_classes=num_classes
                ).to(device)
                clf_rand = build_classifier(
                    input_dim=num_anchors,
                    intermediate_dim=num_anchors,
                    num_classes=num_classes
                ).to(device)
                print(f"Training relative classifier ({metric}, optimized) for baseline {baseline} ...")
                train_classifier(clf_opt, rel_train_opt, train_labels_base,
                                 rel_train_opt, train_labels_base, device,
                                 num_epochs=num_epochs, description=f"Relative_{metric}_Opt")
                print(f"Training relative classifier ({metric}, random) for baseline {baseline} ...")
                train_classifier(clf_rand, rel_train_rand, train_labels_base,
                                 rel_train_rand, train_labels_base, device,
                                 num_epochs=num_epochs, description=f"Relative_{metric}_Rand")
                classifiers_opt[metric] = clf_opt
                classifiers_rand[metric] = clf_rand

            # For each transformer, evaluate the relative classifiers (zero-shot).
            for tname in transformer_names:
                t_feats = features_dict[tname]
                t_test_features = t_feats["test_features"].to(device)
                t_test_labels = t_feats["test_labels"].to(device)
                anch_t_opt = anchor_selector(t_feats["train_features"].to(device))
                # For Mahalanobis, compute transformer-specific inverse covariance:
                t_train_features = t_feats["train_features"].to(device)
                cov_t = compute_covariance_matrix(t_train_features)
                inv_cov_t = torch.linalg.inv(cov_t + 1e-6 * torch.eye(cov_t.size(0), device=cov_t.device))
                for metric, func in proj_funcs.items():
                    # For optimized anchors on transformer data:
                    if metric == "mahalanobis":
                        rel_test_opt = relative_projection_mahalanobis_batched(t_test_features, anch_t_opt, inv_cov_t)
                    else:
                        rel_test_opt = func(t_test_features, anch_t_opt)
                    f1_opt = evaluate_classifier(classifiers_opt[metric], rel_test_opt, t_test_labels, device)
                    results[baseline]["optimized"].setdefault(metric, {}).setdefault(tname, []).append(f1_opt)
                    
                    # For random anchors on transformer data:
                    anchors_t_rand = t_feats["train_features"].to(device)[random_anchor_indices]
                    if metric == "mahalanobis":
                        rel_test_rand = relative_projection_mahalanobis_batched(t_test_features, anchors_t_rand, inv_cov_t)
                    else:
                        rel_test_rand = func(t_test_features, anchors_t_rand)
                    f1_rand = evaluate_classifier(classifiers_rand[metric], rel_test_rand, t_test_labels, device)
                    results[baseline]["random"].setdefault(metric, {}).setdefault(tname, []).append(f1_rand)

    # ----- Compute average scores over seeds -----
    def avg_std(lst):
        return np.mean(lst), np.std(lst)

    print("\n----- Overall Results -----")
    for baseline, metrics in results.items():
        abs_mean, abs_std = avg_std(metrics["absolute"])
        print(f"\nDecoder: {baseline}  |  Absolute Classifier F1: {abs_mean:.2f}% ± {abs_std:.2f}%")
        print("---- Optimized Relative Zero-Shot Results ----")
        header = f"| {'Encoder':<25} | {'Cosine F1 (%)':^15} | {'Euclidean F1 (%)':^17} | {'Mahalanobis F1 (%)':^20} |"
        line = "+" + "-"*27 + "+" + "-"*17 + "+" + "-"*19 + "+" + "-"*22 + "+"
        print(line)
        print(header)
        print(line)
        for tname in transformer_names:
            cos_mean, _ = avg_std(metrics["optimized"]["cosine"][tname])
            euc_mean, _ = avg_std(metrics["optimized"]["euclidean"][tname])
            mah_mean, _ = avg_std(metrics["optimized"]["mahalanobis"][tname])
            print(f"| {tname:<25} | {cos_mean:^15.2f} | {euc_mean:^17.2f} | {mah_mean:^20.2f} |")
        print(line)
        print("---- Random Relative Zero-Shot Results ----")
        print(line)
        print(header)
        print(line)
        for tname in transformer_names:
            cos_mean, _ = avg_std(metrics["random"]["cosine"][tname])
            euc_mean, _ = avg_std(metrics["random"]["euclidean"][tname])
            mah_mean, _ = avg_std(metrics["random"]["mahalanobis"][tname])
            print(f"| {tname:<25} | {cos_mean:^15.2f} | {euc_mean:^17.2f} | {mah_mean:^20.2f} |")
        print(line)

if __name__ == "__main__":
    main()