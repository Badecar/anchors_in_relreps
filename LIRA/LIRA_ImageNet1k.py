import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import timm
from timm.data import resolve_data_config, create_transform
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Import weight optimization routine and random seed utility.
from anchors import optimize_weights
from utils import set_random_seeds

# ----- Covariance and Mahalanobis projection utilities -----
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

# ----- Utility to load a subset of ImageNet1k (for training anchors) -----
def get_imagenet_subset(split, total_samples, seed=42):
    dataset = load_dataset("imagenet-1k", split=split, trust_remote_code=True)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = indices[:total_samples]
    dataset = dataset.select(selected)
    return dataset

# ----- Data collate function for ImageNet1k using only evaluation transforms -----
def collate_fn(batch, transform):
    # Convert to RGB and apply only deterministic (resize/normalization) transforms.
    images = [transform(sample["image"].convert("RGB")) for sample in batch]
    labels = [sample["label"] for sample in batch]
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels

# ----- Get image and text embeddings for a dataset of image-label pairs -----
def extract_image_text_embeddings(dataset, image_encoder, text_encoder, transform, batch_size=64, device="cpu", class_names=None):
    image_encodings = []
    text_encodings = []
    labels_all = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, transform))
    for images, labels in tqdm(loader, desc="Extracting embeddings"):
        images = images.to(device)
        with torch.no_grad():
            img_emb = image_encoder(images)
        image_encodings.append(img_emb.cpu())
        labels_all.append(labels)
        prompts = []
        for lab in labels:
            if class_names is not None:
                label_name = class_names[lab]
            else:
                label_name = str(lab.item())
            prompts.append(f"A photo of a {label_name}.")
        with torch.no_grad():
            # Text encoder is already on GPU.
            txt_emb = text_encoder.encode(prompts, convert_to_tensor=True)
        text_encodings.append(txt_emb.cpu())
    image_encodings = torch.cat(image_encodings, dim=0)
    text_encodings = torch.cat(text_encodings, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    return image_encodings, text_encodings, labels_all

# ----- KMeans-based anchor extraction with weight optimization -----
def get_kmeans_anchors_clustered(src_emb_np, tgt_emb_np, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
    N = src_emb_np.shape[0]
    global_idx = np.arange(N)
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(src_emb_np)
    centers = kmeans.cluster_centers_
    
    clusters_info = []
    anchors_src = []
    anchors_tgt = []
    
    for center in tqdm(centers, desc="Processing clusters"):
        dists = np.linalg.norm(src_emb_np - center, axis=1)
        candidate_order = np.argsort(dists)[:n_closest]
        candidate_global = global_idx[candidate_order]
        candidate_points = src_emb_np[candidate_order]
        weights = optimize_weights(center, candidate_points)
        clusters_info.append((candidate_global, weights, center))
        anchor_src = np.average(src_emb_np[candidate_global], axis=0, weights=weights)
        anchor_tgt = np.average(tgt_emb_np[candidate_global], axis=0, weights=weights)
        anchors_src.append(anchor_src)
        anchors_tgt.append(anchor_tgt)
    
    anchors_src = np.vstack(anchors_src)
    anchors_tgt = np.vstack(anchors_tgt)
    
    if verbose:
        print("KMeans center (first):", centers[0])
        print("First cluster candidate indices:", clusters_info[0][0])
        print("First image anchor:", anchors_src[0])
        print("First text anchor:", anchors_tgt[0])
    
    return anchors_src, anchors_tgt, clusters_info

# ----- Main routine -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seeds(42)

    ###############
    # TRAINING PHASE: Build anchors from 50k paired (image, text) samples
    ###############
    total_train = 50000
    print(f"Loading {total_train} random training images from ImageNet1k...")
    train_dataset = get_imagenet_subset("train", total_train, seed=42)
    class_names = train_dataset.features["label"].names

    # Use the evaluation transform (resize and normalization) only.
    dummy_model = timm.create_model("resnet50", pretrained=True, num_classes=0)
    config = resolve_data_config({}, model=dummy_model)
    transform = create_transform(**config)
    dummy_model.to(device)
    dummy_model.eval()

    # Initialize text encoder on GPU.
    text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    text_encoder.to(device)

    # Cache training embeddings if not already saved.
    emb_cache_file = os.path.join(os.path.dirname(__file__), "train_img_txt_emb.pt")
    if os.path.exists(emb_cache_file):
        print("Loading cached training embeddings...")
        cache = torch.load(emb_cache_file)
        img_emb_train = cache["img_emb_train"]
        txt_emb_train = cache["txt_emb_train"]
    else:
        print("Extracting embeddings for training set (50k samples)...")
        img_emb_train, txt_emb_train, _ = extract_image_text_embeddings(
            train_dataset,
            image_encoder=dummy_model,
            text_encoder=text_encoder,
            transform=transform,
            batch_size=64,
            device=device,
            class_names=class_names
        )
        torch.save({"img_emb_train": img_emb_train, "txt_emb_train": txt_emb_train}, emb_cache_file)
        print(f"Training embeddings saved to {emb_cache_file}")

    img_emb_np = img_emb_train.numpy()
    txt_emb_np = txt_emb_train.numpy()

    # Run KMeans to extract anchors.
    anchor_num = 1025       # Adjust as desired.
    n_closest = 100
    print(f"Running KMeans clustering with {anchor_num} clusters on image embeddings...")
    anchors_img_np, anchors_txt_np, clusters_info = get_kmeans_anchors_clustered(
        src_emb_np=img_emb_np,
        tgt_emb_np=txt_emb_np,
        anchor_num=anchor_num,
        n_closest=n_closest,
        kmeans_seed=42,
        verbose=True
    )
    anchors_img = torch.tensor(anchors_img_np, device=device, dtype=img_emb_train.dtype)
    anchors_txt = torch.tensor(anchors_txt_np, device=device, dtype=txt_emb_train.dtype)

    ###############
    # TESTING PHASE: Evaluate on ImageNet1k validation set using class name prompts
    ###############
    # Use validation data (with only deterministic transforms)
    print("Loading ImageNet1k validation set...")
    val_dataset = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
    val_cache_file = os.path.join(os.path.dirname(__file__), "val_image_emb.pt")
    if os.path.exists(val_cache_file):
        print("Loading cached validation image embeddings...")
        val_image_emb = torch.load(val_cache_file)
    else:
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=lambda b: collate_fn(b, transform))
        print("Extracting image embeddings for validation set...")
        image_emb_list = []
        val_labels_list = []
        for images, labels in tqdm(val_loader, desc="Validation image embeddings"):
            images = images.to(device)
            with torch.no_grad():
                img_out = dummy_model(images)
            image_emb_list.append(img_out.cpu())
            val_labels_list.append(labels)
        val_image_emb = torch.cat(image_emb_list, dim=0)
        torch.save(val_image_emb, val_cache_file)
        print(f"Validation embeddings saved to {val_cache_file}")
    val_labels = torch.tensor(val_dataset["label"])

    # Create (and cache) unique text prompt embeddings for the 1,000 official class names.
    prompt_cache_file = os.path.join(os.path.dirname(__file__), "txt_prompt_emb.pt")
    imagenet_class_names = val_dataset.features["label"].names
    if os.path.exists(prompt_cache_file):
        print("Loading cached text prompt embeddings...")
        txt_prompt_emb = torch.load(prompt_cache_file)
    else:
        prompts = [f"A photo of a {label}." for label in imagenet_class_names]
        print("Computing text embeddings for class prompts...")
        with torch.no_grad():
            txt_prompt_emb = text_encoder.encode(prompts, convert_to_tensor=True).to(device)
        torch.save(txt_prompt_emb, prompt_cache_file)
        print(f"Text prompt embeddings saved to {prompt_cache_file}")

    # Compute covariance matrices and inverse covariances for the Mahalanobis projection.
    # For the images:
    cov_img = compute_covariance_matrix(val_image_emb.to(device))
    inv_cov_img = torch.linalg.inv(cov_img + 1e-6 * torch.eye(cov_img.size(0), device=device))
    img_rel = relative_projection_mahalanobis_batched(val_image_emb.to(device), anchors_img, inv_cov_img)

    # For the text prompts:
    cov_text = compute_covariance_matrix(txt_prompt_emb)
    inv_cov_text = torch.linalg.inv(cov_text + 1e-6 * torch.eye(cov_text.size(0), device=device))
    txt_rel = relative_projection_mahalanobis_batched(txt_prompt_emb, anchors_txt, inv_cov_text)

    # Now compute pairwise Mahalanobis-based distances between image relreps and text relreps.
    # We first compute a covariance over the text relative representations.
    cov_rel = compute_covariance_matrix(txt_rel.to(device))
    inv_cov_rel = torch.linalg.inv(cov_rel + 1e-6 * torch.eye(cov_rel.size(0), device=device))
    
    print("Evaluating image classification using Mahalanobis-based relative representations ...")
    top1_correct = 0
    top5_correct = 0
    total = img_rel.size(0)
    batch_size = 64
    for i in range(0, total, batch_size):
        batch_img_rel = img_rel[i:i+batch_size]
        # Compute distances between batch image relreps and all text prompt relreps.
        dists = relative_projection_mahalanobis_batched(batch_img_rel, txt_rel, inv_cov_rel)
        # Higher similarity corresponds to less distance.
        sim = dists  # Already negative distances.
        top1 = sim.argmax(dim=1)
        top5 = sim.topk(5, dim=1)[1]
        batch_labels = val_labels[i:i+batch_size].to(device)
        top1_correct += (top1 == batch_labels).sum().item()
        for j in range(sim.size(0)):
            if batch_labels[j] in top5[j]:
                top5_correct += 1

    top1_acc = 100 * top1_correct / total
    top5_acc = 100 * top5_correct / total
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

if __name__ == "__main__":
    main()