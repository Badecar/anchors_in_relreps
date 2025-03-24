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
    "vit_base_resnet50_384",
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
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # PARAMETERS
    train_perc = 1.0
    fine_grained = False
    target_key = "coarse_label" if not fine_grained else "fine_label"
    num_anchors = 768
    num_epochs = 5
    batch_size = 32
    coverage_w = 0.90  # Weight for coverage
    diversity_w = 1 - coverage_w  # Weight for diversity

    print("Loading CIFAR-100 dataset...")
    train_dataset = get_dataset("train", perc=train_perc)
    test_dataset = get_dataset("test", perc=train_perc)
    if hasattr(train_dataset.features[target_key], "num_classes"):
        num_classes = train_dataset.features[target_key].num_classes
    else:
        num_classes = 20 if not fine_grained else 100

    #####################################
    # 1. Train the anchors using vit_base_resnet50_384 only.
    #####################################
    anchor_model_name = "vit_base_patch16_224"
    print(f"\n----- Training anchors using {anchor_model_name} -----")
    anchor_model = timm.create_model(anchor_model_name, pretrained=True, num_classes=0)
    anchor_model.to(device)
    anchor_model.eval()
    config_anchor = resolve_data_config({}, model=anchor_model)
    transform_anchor = create_transform(**config_anchor)

    train_loader_anchor = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, transform_anchor)
    )
    test_loader_anchor = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, transform_anchor)
    )

    print("Extracting anchor training features...")
    train_features_anchor, _ = extract_features(anchor_model, train_loader_anchor, device)
    print("Extracting anchor test features...")
    test_features_anchor, _ = extract_features(anchor_model, test_loader_anchor, device)
    hidden_dim_anchor = train_features_anchor.shape[1]
    print("Anchor feature dimension:", hidden_dim_anchor)

    # Train (optimize) anchors using the anchor model's training features.
    anchor_selector, _ = get_optimized_anchors(
        emb=[train_features_anchor.cpu().numpy()],
        anchor_num=num_anchors,
        epochs=50,            # Adjust as needed
        lr=1e-1,
        coverage_weight=coverage_w,
        diversity_weight=diversity_w,
        exponent=1,
        verbose=True,
        device=device
    )
    for param in anchor_selector.parameters():
        param.requires_grad = False

    # Compute fixed anchors from the anchor model.
    # These fixed anchors will be used for all subsequent relative projections.
    anchors_fixed = anchor_selector(train_features_anchor.to(device))

    #####################################
    # 2. Loop over each baseline model.
    #####################################
    results = dict()
    for baseline_name in baseline_transformer_names:
        print(f"\n----- Baseline Transformer: {baseline_name} -----")
        baseline_model = timm.create_model(baseline_name, pretrained=True, num_classes=0)
        baseline_model.to(device)
        baseline_model.eval()
        config_base = resolve_data_config({}, model=baseline_model)
        transform_base = create_transform(**config_base)
        train_loader_base = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, transform_base)
        )
        test_loader_base = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, transform_base)
        )
        print("Extracting baseline training features...")
        train_features_base, train_labels_base = extract_features(baseline_model, train_loader_base, device)
        print("Extracting baseline test features...")
        test_features_base, test_labels_base = extract_features(baseline_model, test_loader_base, device)
        hidden_dim_base = train_features_base.shape[1]
        print("Baseline feature dimension:", hidden_dim_base)

        anch = anchor_selector(train_features_base.to(device))

        # IMPORTANT: Instead of computing anchors from baseline features, we use the fixed anchors.
        train_rel = relative_projection(train_features_base, anch)
        test_rel = relative_projection(test_features_base, anch)

        # Build classifiers on the baseline model's features.
        abs_classifier = build_classifier(input_dim=hidden_dim_base, intermediate_dim=num_anchors, num_classes=num_classes).to(device)
        rel_classifier = build_classifier(input_dim=num_anchors, intermediate_dim=num_anchors, num_classes=num_classes).to(device)
        print("\nTraining baseline absolute classifier...")
        train_classifier(abs_classifier, train_features_base, train_labels_base, test_features_base, test_labels_base, device,
                         num_epochs=num_epochs, description="Absolute")
        print("Training baseline relative classifier...")
        train_classifier(rel_classifier, train_rel, train_labels_base, test_rel, test_labels_base, device,
                         num_epochs=num_epochs, description="Relative")

        results[baseline_name] = {"relative": {}}

        #####################################
        # 3. For each transformer (testing), extract train features,
        #    compute anchors using the pretrained anchor selector,
        #    and then evaluate relative performance.
        #####################################
        for tname in transformer_names:
            print(f"\nEvaluating transformer: {tname}")
            test_model = timm.create_model(tname, pretrained=True, num_classes=0)
            test_model.to(device)
            test_model.eval()
            # Create the proper transform for the current transformer.
            config_t = resolve_data_config({}, model=test_model)
            transform_t = create_transform(**config_t)
            
            # Extract transformer train features.
            train_loader_t = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, transform_t)
            )
            t_train_features, _ = extract_features(test_model, train_loader_t, device)
            
            # Compute anchors from the transformer train features using the pretrained anchor selector.
            t_anchors = anchor_selector(t_train_features.to(device))
            
            # Extract transformer test features.
            test_loader_t = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, transform_t)
            )
            t_test_features, _ = extract_features(test_model, test_loader_t, device)
            
            # Compute the relative projection using the computed anchors.
            t_test_rel = relative_projection(t_test_features, t_anchors)
            rel_acc = evaluate_classifier(rel_classifier, t_test_rel, test_labels_base, device)
            print(f"Transformer {tname} | Relative Test Accuracy: {rel_acc:.2f}%")
            results[baseline_name]["relative"][tname] = rel_acc

    #####################################
    # 4. Report overall results.
    #####################################
    print("\n----- Overall Results -----")
    for baseline_name, metrics in results.items():
        print(f"\nDecoder: {baseline_name}")
        print("Relative classifier results:")
        for tname, acc in metrics["relative"].items():
            print(f"  - {tname}: {acc:.2f}%")

if __name__ == "__main__":
    main()
