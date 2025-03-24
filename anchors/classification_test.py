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
from utils import set_random_seeds
from P_anchors import get_optimized_anchors, AnchorSelector
original_forward = AnchorSelector.forward
def patched_forward(self, X):
    result = original_forward(self, X)
    # If the forward returns a tuple, use only the first element.
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

# Computes the relative projection.
def relative_projection(x, anchors):
    # x: (batch, feature_dim) and anchors: (num_anchors, feature_dim)
    # Returns relative representations of shape (batch, num_anchors)
    x = F.normalize(x, p=2, dim=-1)
    anchors = F.normalize(anchors, p=2, dim=-1).to(x.device)
    anchors = anchors.to(x.device)  # Ensure anchors and x are on the same device.
    # torch.einsum returns shape (num_anchors, batch) so we transpose.
    return torch.einsum("bm, am -> ba", x, anchors)

# Get the dataset (with an optional percentage of samples)
def get_dataset(split: str, perc: float = 1.0):
    dataset = load_dataset("cifar100", split=split)
    if perc < 1.0:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: int(len(dataset) * perc)]
        dataset = dataset.select(indices)
    return dataset

# Given a batch from the HF dataset, apply the transform on each sample's image.
def collate_fn(batch, transform):
    images = [transform(sample["img"]) for sample in batch]
    # Use coarse_label (or change to fine_label if needed)
    labels = [sample["coarse_label"] for sample in batch]
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

# Extract features in batches using the pretrained encoder.
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

# Build a simple classifier model.
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

# Train a classifier on precomputed features.
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
        acc = evaluate_classifier(classifier, test_feats, test_labels, device)
        print(f"[{description}] Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Test Acc: {acc:.2f}%")

# Evaluate classifier accuracy.
def evaluate_classifier(classifier, feats, labels, device):
    classifier.eval()
    with torch.no_grad():
        logits = classifier(feats.to(device))
        preds = torch.argmax(logits, dim=1).cpu()
    correct = (preds == labels).sum().item()
    acc = 100 * correct / labels.size(0)
    return acc

# Lists of transformer model names.
transformer_names = [
    "rexnet_100",
    "vit_base_patch16_224",
    "vit_base_resnet50_384", # Takes a long time to run
    "vit_small_patch16_224"
]
baseline_transformer_names = [
    "rexnet_100",
    "vit_base_patch16_224",
    "vit_base_resnet50_384",
    "vit_small_patch16_224"
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(42)
    
    # PARAMETERS
    train_perc = 1.0
    fine_grained = False
    target_key = "coarse_label" if not fine_grained else "fine_label"
    num_anchors = 200
    num_epochs = 5
    batch_size = 32
    coverage_w = 1
    diversity_w = 1 - coverage_w

    print("Loading CIFAR-100 dataset...")
    train_dataset = get_dataset("train", perc=train_perc)
    test_dataset = get_dataset("test", perc=train_perc)
    if hasattr(train_dataset.features[target_key], "num_classes"):
        num_classes = train_dataset.features[target_key].num_classes
    else:
        num_classes = 20 if not fine_grained else 100

    #####################################
    # Precompute features for all models (union of baseline and transformer names)
    #####################################
    all_model_names = set(baseline_transformer_names + transformer_names)
    print(all_model_names)
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

    # Compute a fixed set of random anchor indices (same for all models)
    # All models' training features have the same number of samples.
    sample_count = features_dict[list(features_dict.keys())[0]]["train_features"].shape[0]
    random_anchor_indices = np.sort(np.random.choice(sample_count, num_anchors, replace=False))

    #####################################
    # 1. Train the anchors using the anchor model (vit_small_patch16_224 if available) and its precomputed features.
    #####################################
    if "vit_small_patch16_224" in features_dict:
        anchor_model_name = "vit_small_patch16_224"
        print("Using vit_small_patch16_224 as the anchor model.")
    else:
        anchor_model_name = "rexnet_100"
        print("Using rexnet_100 as the anchor model.")
    print(f"\n----- Training optimized anchors using {anchor_model_name} -----")
    anchor_train_feats = features_dict[anchor_model_name]["train_features"]
    anchor_selector, _ = get_optimized_anchors(
        emb=[anchor_train_feats.cpu().numpy()],
        anchor_num=num_anchors,
        epochs=200,            # Adjust as needed
        lr=1e-3,
        coverage_weight=coverage_w,
        diversity_weight=diversity_w,
        exponent=1,
        verbose=True,
        device=device
    )
    for param in anchor_selector.parameters():
        param.requires_grad = False

    #####################################
    # 2. Loop over each baseline model.
    #####################################
    results = dict()
    for baseline_name in baseline_transformer_names:
        print(f"\n----- Baseline Transformer: {baseline_name} -----")
        base_feats = features_dict[baseline_name]
        train_features_base, train_labels_base = base_feats["train_features"], base_feats["train_labels"]
        test_features_base, test_labels_base = base_feats["test_features"], base_feats["test_labels"]

        # Optimized anchors for baseline training features.
        anch_baseline = anchor_selector(train_features_base.to(device))
        train_rel_optimized = relative_projection(train_features_base, anch_baseline)

        # Random anchors using fixed indices from train_features.
        anchors_random = train_features_base[random_anchor_indices]
        train_rel_random = relative_projection(train_features_base, anchors_random)

        # Train absolute classifier (on original features)
        abs_classifier = build_classifier(
            input_dim=train_features_base.shape[1],
            intermediate_dim=num_anchors,
            num_classes=num_classes
        ).to(device)
        print("\nTraining baseline absolute classifier...")
        train_classifier(
            abs_classifier, train_features_base, train_labels_base,
            test_features_base, test_labels_base, device,
            num_epochs=num_epochs, description="Absolute"
        )
        abs_acc = evaluate_classifier(abs_classifier, test_features_base, test_labels_base, device)

        # Train relative classifier with optimized anchors.
        P_rel_classifier = build_classifier(
            input_dim=num_anchors,
            intermediate_dim=num_anchors,
            num_classes=num_classes
        ).to(device)
        print("Training baseline optimized relative classifier...")
        train_classifier(
            P_rel_classifier, train_rel_optimized, train_labels_base,
            train_rel_optimized, train_labels_base, device,
            num_epochs=num_epochs, description="Relative_Optimized"
        )

        # Train relative classifier with random anchors.
        rand_rel_classifier = build_classifier(
            input_dim=num_anchors,
            intermediate_dim=num_anchors,
            num_classes=num_classes
        ).to(device)
        print("Training baseline random relative classifier...")
        train_classifier(
            rand_rel_classifier, train_rel_random, train_labels_base,
            train_rel_random, train_labels_base, device,
            num_epochs=num_epochs, description="Relative_Random"
        )

        results[baseline_name] = {"absolute": abs_acc, "optimized": {}, "random": {}}

        #####################################
        # 3. For each transformer, use their precomputed test features.
        #####################################
        for tname in transformer_names:
            print(f"\nEvaluating transformer: {tname}")
            t_feats = features_dict[tname]
            t_train_features, _ = t_feats["train_features"], t_feats["train_labels"]
            t_test_features, t_test_labels = t_feats["test_features"], t_feats["test_labels"]

            # For optimized anchors on transformer data:
            anch_t_opt = anchor_selector(t_train_features.to(device))
            t_test_rel_opt = relative_projection(t_test_features, anch_t_opt)
            rel_acc_opt = evaluate_classifier(P_rel_classifier, t_test_rel_opt, t_test_labels, device)
            print(f"Transformer {tname} | Optimized Relative Test Accuracy: {rel_acc_opt:.2f}%")

            # For random anchors on transformer data (using the same random indices):
            anchors_t_rand = t_train_features[random_anchor_indices]
            t_test_rel_rand = relative_projection(t_test_features, anchors_t_rand)
            rel_acc_rand = evaluate_classifier(rand_rel_classifier, t_test_rel_rand, t_test_labels, device)
            print(f"Transformer {tname} | Random Relative Test Accuracy: {rel_acc_rand:.2f}%")

            results[baseline_name]["optimized"][tname] = rel_acc_opt
            results[baseline_name]["random"][tname] = rel_acc_rand

    #####################################
    # 4. Report overall results.
    #####################################
    print("\n----- Overall Results -----")
    for baseline_name, metrics in results.items():
        print(f"\nDecoder: {baseline_name}")
        print(f"Absolute Classifier Accuracy: {metrics['absolute']:.2f}%")
        print("Optimized Relative Classifier results:")
        for tname, acc in metrics["optimized"].items():
            print(f"  - {tname}: {acc:.2f}%")
        print("Random Relative Classifier results:")
        for tname, acc in metrics["random"].items():
            print(f"  - {tname}: {acc:.2f}%")

if __name__ == "__main__":
    main()