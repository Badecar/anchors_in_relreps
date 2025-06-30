import os
import torch
import timm
import random
from tqdm import tqdm
from torch.utils.data import DataLoader

from datasets import load_dataset
from timm.data import resolve_data_config, create_transform

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

def get_dataset(split, perc=1.0, dataset_name="CIFAR100_coarse"):
    if dataset_name.lower() == "imagenet1k":
        dataset = load_dataset("imagenet-1k", split=split, trust_remote_code=True)
    else:
        dataset = load_dataset("cifar100", split=split)
    if perc < 1.0:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[: int(len(dataset) * perc)]
        dataset = dataset.select(indices)
    return dataset

def collate_fn(batch, transform, dataset_name="CIFAR100_coarse"):
    if dataset_name.lower() == "imagenet1k":
        images = [transform(sample["image"]) for sample in batch]
        labels = [sample["label"] for sample in batch]
    else:
        images = [transform(sample["img"]) for sample in batch]
        labels = [sample["coarse_label"] for sample in batch]
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

def get_features_dict(dataset_name, train_perc, batch_size, device, transformer_names):
    print(f"Loading {dataset_name} dataset...")
    train_dataset = get_dataset("train", perc=train_perc, dataset_name=dataset_name)
    test_dataset = get_dataset("test", perc=train_perc, dataset_name=dataset_name)
    if dataset_name.lower() == "imagenet1k":
        num_classes = 1000
    else:
        if hasattr(train_dataset.features["coarse_label"], "num_classes"):
            num_classes = train_dataset.features["coarse_label"].num_classes
        else:
            num_classes = 20

    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_path)
    features_file = os.path.join(parent_dir, "datasets", "cifar", f"features_dict_{dataset_name}.pt")

    if os.path.exists(features_file):
        features_dict = torch.load(features_file)
        print(f"Loaded precomputed features from {features_file}")
    else:
        features_dict = {}
        for model_name in transformer_names:
            print(f"\nExtracting features for model: {model_name}")
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            model.to(device)
            model.eval()
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)
            # Pass dataset_name to the collate_fn so the correct keys are used.
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                        collate_fn=lambda batch: collate_fn(batch, transform, dataset_name))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                        collate_fn=lambda batch: collate_fn(batch, transform, dataset_name))
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

    return features_dict, num_classes