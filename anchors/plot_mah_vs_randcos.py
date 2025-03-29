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

# Assume these functions are defined in the project
from utils import set_random_seeds
from P_anchors import get_optimized_anchors, AnchorSelector

# Patch AnchorSelector.forward as in the original script
original_forward = AnchorSelector.forward
def patched_forward(self, X):
    result = original_forward(self, X)
    if isinstance(result, tuple):
        return result[0]
    return result
AnchorSelector.forward = patched_forward

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

def relative_projection_mahalanobis_batched(x, anchors, inv_cov, batch_size=1024):
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)  # (B, num_anchors, d)
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        result.append(-dists)
    return torch.cat(result, dim=0)

def compute_covariance_matrix(features):
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    cov = (centered.t() @ centered) / (features.size(0) - 1)
    return cov

# ----- Classifier and training/evaluation functions -----
def build_classifier(input_dim, intermediate_dim, num_classes):
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, intermediate_dim),
        nn.Tanh(),
        Lambda(lambda x: x.permute(1, 0)),
        nn.InstanceNorm1d(intermediate_dim),
        Lambda(lambda x: x.permute(1, 0)),
        nn.Linear(intermediate_dim, num_classes)
    )

def train_classifier(classifier, train_feats, train_labels, device, num_epochs=7):
    classifier.train()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
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

# ----- Main experiment -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_perc = 1.0
    batch_size = 64
    num_epochs = 7
    n_seeds = 10  # seeds per anchor configuration

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
        for model_name in ["vit_base_resnet50_384", "vit_small_patch16_224"]:
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
    
    # For training the relative decoder, use features from vit_base_resnet50_384.
    train_feats_train = features_dict["vit_base_resnet50_384"]["train_features"]
    train_labels_train = features_dict["vit_base_resnet50_384"]["train_labels"]
    
    # For evaluation (zero-shot), use features from vit_small_patch16_224.
    train_feats_test = features_dict["vit_small_patch16_224"]["train_features"]
    test_feats_test = features_dict["vit_small_patch16_224"]["test_features"]
    test_labels_test = features_dict["vit_small_patch16_224"]["test_labels"]
    
    # --- Experiment: Loop over anchor numbers ---
    anchor_nums = list(range(100, 1001, 50))
    results_random = {a: [] for a in anchor_nums}
    results_optimized = {a: [] for a in anchor_nums}
    
    # For each configuration, we use the same random anchor indices (fixed across models)
    for num_anchors in anchor_nums:
        print(f"\nEvaluating for {num_anchors} anchors...")
        for seed in range(42, 42 + n_seeds):
            print(f" Seed {seed}")
            set_random_seeds(seed)
            sample_count = train_feats_train.shape[0]
            # FIXED RANDOM ANCHORS: compute once and reuse for both models
            random_indices = np.sort(np.random.choice(sample_count, num_anchors, replace=False))
            
            # -------- Training phase (vit_base_resnet50_384 features) --------
            # Random method: use the same fixed indices for training features
            anchors_random_train = train_feats_train[random_indices].to(device)
            rel_train_random = relative_projection_cosine(train_feats_train.to(device), anchors_random_train)
            
            # Optimized method: optimize anchors on training features (vit_base_resnet50_384)
            coverage_w = 0.9
            diversity_w = 1 - coverage_w
            anti_collapse_w = 0
            anchor_selector, _ = get_optimized_anchors(
                emb=[train_feats_train.cpu().numpy()],
                anchor_num=num_anchors,
                epochs=200,
                lr=1e-2,
                coverage_weight=coverage_w,
                diversity_weight=diversity_w,
                anti_collapse_w=anti_collapse_w,
                exponent=1,
                dist_measure="euclidean",
                verbose=False,
                device=device
            )
            for param in anchor_selector.parameters():
                param.requires_grad = False
            anchors_optimized_train = anchor_selector(train_feats_train.to(device))
            cov_train = compute_covariance_matrix(train_feats_train.to(device))
            inv_cov_train = torch.linalg.inv(cov_train + 1e-6 * torch.eye(cov_train.size(0), device=cov_train.device))
            rel_train_optimized = relative_projection_mahalanobis_batched(train_feats_train.to(device),
                                                                         anchors_optimized_train, inv_cov_train)
            # Train classifiers on the relative representations
            clf_random = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            clf_optimized = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            print(" Training Random relative decoder:")
            clf_random = train_classifier(clf_random, rel_train_random, train_labels_train.to(device),
                                          device, num_epochs=num_epochs)
            print(" Training Optimized relative decoder:")
            clf_optimized = train_classifier(clf_optimized, rel_train_optimized, train_labels_train.to(device),
                                             device, num_epochs=num_epochs)
            
            # -------- Testing phase (vit_small_patch16_224 features) --------
            # For the random method, use the SAME fixed random indices on vit_small_patch16_224 training features.
            anchors_random_test = train_feats_test[random_indices].to(device)
            rel_test_random = relative_projection_cosine(test_feats_test.to(device), anchors_random_test)
            f1_random = evaluate_classifier(clf_random, rel_test_random, test_labels_test.to(device), device)
            
            # # For the optimized method, optimize anchors on the vit_small_patch16_224 training features.
            # anchor_selector_test, _ = get_optimized_anchors(
            #     emb=[train_feats_test.cpu().numpy()],
            #     anchor_num=num_anchors,
            #     epochs=200,
            #     lr=1e-2,
            #     coverage_weight=coverage_w,
            #     diversity_weight=diversity_w,
            #     anti_collapse_w=anti_collapse_w,
            #     exponent=1,
            #     dist_measure="euclidean",
            #     verbose=False,
            #     device=device
            # )
            # for param in anchor_selector_test.parameters():
            #     param.requires_grad = False
            anchors_optimized_test = anchor_selector(train_feats_test.to(device))
            cov_test = compute_covariance_matrix(train_feats_test.to(device))
            inv_cov_test = torch.linalg.inv(cov_test + 1e-6 * torch.eye(cov_test.size(0), device=cov_test.device))
            rel_test_optimized = relative_projection_mahalanobis_batched(test_feats_test.to(device),
                                                                        anchors_optimized_test, inv_cov_test)
            f1_optimized = evaluate_classifier(clf_optimized, rel_test_optimized, test_labels_test.to(device), device)
            
            print(f"  Random (cosine) F1: {f1_random:.2f}%, Optimized (mahalanobis) F1: {f1_optimized:.2f}%")
            results_random[num_anchors].append(f1_random)
            results_optimized[num_anchors].append(f1_optimized)
    
    # --- Compute mean and std for each anchor number ---
    anchor_nums_arr = []
    random_means = []
    random_stds = []
    optimized_means = []
    optimized_stds = []
    for a in anchor_nums:
        anchor_nums_arr.append(a)
        r_mean = np.mean(results_random[a])
        r_std = np.std(results_random[a])
        o_mean = np.mean(results_optimized[a])
        o_std = np.std(results_optimized[a])
        random_means.append(r_mean)
        random_stds.append(r_std)
        optimized_means.append(o_mean)
        optimized_stds.append(o_std)
        print(f"Anchors: {a}, Random: {r_mean:.2f}% ± {r_std:.2f}%, Optimized: {o_mean:.2f}% ± {o_std:.2f}%")
    
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
    plt.xlabel("Number of Anchors")
    plt.ylabel("F1 Score (%)")
    plt.title("Relative Decoder Performance vs Number of Anchors")
    plt.legend()
    plt.grid(True)
    plt.show()

    csv_file = os.path.join(current_path, "relative_decoder_results.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Anchors", "Random_mean", "Random_std", "Optimized_mean", "Optimized_std"])
        for a, r_mean, r_std, o_mean, o_std in zip(anchor_nums_arr, random_means, random_stds, optimized_means, optimized_stds):
            writer.writerow([a, r_mean, r_std, o_mean, o_std])
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()
