import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from datasets import load_dataset
from tqdm import tqdm
import csv
import torch.nn.functional as F
import pickle

# Additional imports for plotting
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Import your weight optimization routine (ensure optimize_weights is defined in your project)
from anchors import optimize_weights  # Adjust the import path if necessary

# ----- Helper functions for encoding -----
def encode_images(model, loader):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Image Embeddings"):
            images = images.to(device)
            feats = model(images)
            features.append(feats.cpu())
            torch.cuda.empty_cache()
    return torch.cat(features, dim=0)

def encode_texts(text_encoder, loader, bs=64):
    """
    Gather all captions in order, encode them in batches and return both the tensor and the captions
    """
    captions = []
    with torch.no_grad():
        for _, cap in loader:
            if isinstance(cap, list):
                captions.extend(cap)
            else:
                captions.append(cap)
        outputs = []
        for i in tqdm(range(0, len(captions), bs), desc="Text Embeddings"):
            batch = captions[i:i+bs]
            emb = text_encoder.encode(batch, convert_to_tensor=True)
            outputs.append(emb.cpu())
    return torch.cat(outputs, dim=0), captions

# ----- KMeans anchors extraction -----
def get_kmeans_anchors_clustered(img_emb_np, txt_emb_np, anchor_num, n_closest=20, kmeans_seed=42, verbose=False):
    """
    Run KMeans on the image embeddings and compute weighted anchors.
    For each cluster, select the n_closest image candidates (based on Euclidean distance)
    and use the same indices to compute the corresponding text anchor.
    """
    N = img_emb_np.shape[0]
    global_idx = np.arange(N)
    kmeans = KMeans(n_clusters=anchor_num, random_state=kmeans_seed)
    kmeans.fit(img_emb_np)
    centers = kmeans.cluster_centers_
    
    clusters_info = []
    anchors_img = []
    anchors_txt = []
    
    for center in tqdm(centers, desc="Processing Clusters"):
        dists = np.linalg.norm(img_emb_np - center, axis=1)
        candidate_order = np.argsort(dists)[:n_closest]
        candidate_idxs = global_idx[candidate_order]
        candidate_points = img_emb_np[candidate_order]
        weights = optimize_weights(center, candidate_points)
        clusters_info.append((candidate_idxs, weights, center))
        
        anchor_img = np.average(img_emb_np[candidate_idxs], axis=0, weights=weights)
        anchor_txt = np.average(txt_emb_np[candidate_idxs], axis=0, weights=weights)
        anchors_img.append(anchor_img)
        anchors_txt.append(anchor_txt)
    
    anchors_img = np.vstack(anchors_img)
    anchors_txt = np.vstack(anchors_txt)
    
    if verbose:
        print("First cluster center:", centers[0])
        print("Candidate indices for first cluster:", clusters_info[0][0])
        print("Image anchor (first):", anchors_img[0])
        print("Text anchor (first):", anchors_txt[0])
        
    return anchors_img, anchors_txt, clusters_info

# ----- Function to save image-caption mapping to CSV -----
def save_captions_csv(dataset, csv_filename="sbu_captions.csv"):
    """
    Save the mapping from image filenames to captions.
    Only works if the dataset is in raw mode (i.e. not using an existing CSV).
    """
    if dataset.use_csv:
        print("Dataset is already using a CSV file; not overwriting.")
        return
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_filename", "caption"])
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample is None:
                continue
            _, caption = sample
            image_filename = f"{idx}.jpg"  
            writer.writerow([image_filename, caption])
    print(f"Captions saved to {csv_filename}")

# ----- Identity module to remove classification head from ResNet-50 -----
class Identity(nn.Module):
    def forward(self, x):
        return x

# ----- Set random seeds for reproducibility -----
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_random_seeds(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Custom SBU Captions Dataset -----
class SBUDataset(Dataset):
    """
    A dataset wrapper for SBU Captions.
    In "raw" mode (if no CSV exists) the dataset downloads images using keys such as "image", "img_url", "img", or "image_url" 
    and returns the image (after transformation) with its caption.
    If any download error occurs, the sample is skipped (returns None).
    If a CSV file (default "sbu_captions.csv") exists, it uses that mapping of image filename → caption.
    """
    def __init__(self, split="train", transform=None, max_samples=10000, csv_filename="sbu_captions.csv"):
        self.image_folder = "datasets/sbu/sbu_images"
        os.makedirs(self.image_folder, exist_ok=True)
        self.transform = transform
        self.csv_filename = csv_filename

        if os.path.exists(csv_filename):
            print(f"Found {csv_filename}. Loading image-caption mapping from CSV...")
            self.caption_map = {}
            with open(csv_filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.caption_map[row["image_filename"]] = row["caption"]
            self.filenames = sorted(self.caption_map.keys())
            self.use_csv = True
        else:
            ds = load_dataset("sbu_captions", split=split)
            self.samples = ds.select(range(min(max_samples, len(ds))))
            self.use_csv = False

    def __len__(self):
        return len(self.filenames) if self.use_csv else len(self.samples)

    def __getitem__(self, idx):
        if self.use_csv:
            filename = self.filenames[idx]
            image_filename = os.path.basename(filename)
            image_path = os.path.join(self.image_folder, image_filename)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return None
            if self.transform:
                image = self.transform(image)
            caption = self.caption_map[filename]
            return image, caption
        else:
            sample = self.samples[idx]
            possible_keys = ["image", "img_url", "img", "image_url"]
            img_url = None
            for key in possible_keys:
                if key in sample:
                    img_url = sample[key]
                    break
            if img_url is None:
                raise KeyError(f"No valid image URL found in sample. Available keys: {list(sample.keys())}")
            
            image_filename = f"{idx}.jpg"
            image_path = os.path.join(self.image_folder, image_filename)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                try:
                    response = requests.get(img_url, stream=True)
                    response.raise_for_status()
                    image = Image.open(response.raw).convert("RGB")
                    image.save(image_path)
                except Exception as e:
                    print(f"Skipping sample {idx} due to error: {e}")
                    return None
            if self.transform:
                image = self.transform(image)
            caption = sample["caption"]
            return image, caption

# Custom collate function.
def custom_collate_fn(batch):
    filtered = [item for item in batch if item is not None]
    if len(filtered) == 0:
        return None
    images, captions = zip(*filtered)
    return torch.stack(images), list(captions)

# ----- Additional helper functions for relative representations -----
def compute_covariance_matrix(features):
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    cov = (centered.t() @ centered) / (features.size(0) - 1)
    return cov

def relative_projection_mahalanobis_batched(x, anchors, inv_cov, batch_size=512):
    """
    Projects x (N×D) relative to anchors (K×D) using Mahalanobis distance.
    Returns a new representation [N, K] given by negative distances.
    """
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, device=x.device, dtype=x.dtype)
    result = []
    for i in range(0, x.size(0), batch_size):
        x_batch = x[i:i+batch_size]  # [bs, D]
        diff = x_batch.unsqueeze(1) - anchors.unsqueeze(0)  # [bs, K, D]
        dists = torch.sqrt(torch.einsum("bij,jk,bik->bi", diff, inv_cov, diff) + 1e-8)
        result.append(-dists)  # negative distances as similarities
    return torch.cat(result, dim=0)

def relative_projection_cosine(x, anchors):
    # Using cosine similarity as representation.
    x_norm = F.normalize(x, p=2, dim=-1)
    anchors_norm = F.normalize(anchors, p=2, dim=-1)
    return torch.matmul(x_norm, anchors_norm.t())

def main():
    # ---- Options to choose anchors and similarity metric ----
    anchor_method = "kmeans"       # Options: "kmeans" or "random"
    dist_metric = "mahalanobis"         # Options: "mahalanobis" or "cosine"
    plot_spaces = True             # Set to True to plot the latent and relative spaces using PCA

    preprocess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SBUDataset(split="train", transform=preprocess_img, max_samples=10000, csv_filename="datasets/sbu/sbu_captions.csv")
    if not dataset.use_csv:
        print("Saving captions mapping to CSV...")
        save_captions_csv(dataset, csv_filename="sbu_captions.csv")
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    
    # Build encoders
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = Identity()
    resnet = resnet.to(device)
    resnet.eval()
    
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # --- Load or Compute embeddings ---
    embeddings_dir = os.path.dirname(os.path.abspath(__file__))
    img_embeds_file = os.path.join(embeddings_dir, "img_embeds.pt")
    txt_embeds_file = os.path.join(embeddings_dir, "txt_embeds.pt")
    captions_file = os.path.join(embeddings_dir, "captions_list.pkl")
    
    if os.path.exists(img_embeds_file) and os.path.exists(txt_embeds_file) and os.path.exists(captions_file):
        print("Loading saved embeddings and captions...")
        img_embeds = torch.load(img_embeds_file)
        txt_embeds = torch.load(txt_embeds_file)
        with open(captions_file, "rb") as f:
            captions_list = pickle.load(f)
    else:
        print("Extracting image embeddings...")
        img_embeds = encode_images(resnet, loader)
        print("Extracting text embeddings...")
        txt_embeds, captions_list = encode_texts(text_encoder, loader)
        print(f"Extracted embeddings for {img_embeds.size(0)} samples.")
        torch.save(img_embeds, img_embeds_file)
        torch.save(txt_embeds, txt_embeds_file)
        with open(captions_file, "wb") as f:
            pickle.dump(captions_list, f)
        print("Embeddings and captions saved for later use.")
    
    total = img_embeds.size(0)

    # --- Compute global anchors ---
    num_anchors = 800
    n_closest = 20
    if anchor_method == "kmeans":
        print("Computing global KMeans anchors for the entire dataset...")
        global_anchors_img, global_anchors_txt, _ = get_kmeans_anchors_clustered(
            img_embeds.numpy(), txt_embeds.numpy(), anchor_num=num_anchors, n_closest=n_closest, kmeans_seed=42, verbose=True
        )
    elif anchor_method == "random":
        print("Sampling global anchors randomly from the embeddings...")
        random_indices = np.sort(np.random.choice(total, num_anchors, replace=False))
        global_anchors_img = img_embeds[random_indices].numpy()
        global_anchors_txt = txt_embeds[random_indices].numpy()
    else:
        raise ValueError("Invalid anchor_method. Choose 'kmeans' or 'random'.")
    
    # --- Compute covariance matrices if needed ---
    if dist_metric == "mahalanobis":
        global_cov_img = compute_covariance_matrix(img_embeds)
        d_img = global_cov_img.size(0)
        global_inv_cov_img = torch.linalg.inv(global_cov_img + 1e-6 * torch.eye(d_img, device=img_embeds.device))
    
        global_cov_txt = compute_covariance_matrix(txt_embeds)
        d_txt = global_cov_txt.size(0)
        global_inv_cov_txt = torch.linalg.inv(global_cov_txt + 1e-6 * torch.eye(d_txt, device=txt_embeds.device))
    
    # --- Sample 500 pairs ---
    if total < 500:
        print("Not enough samples to perform evaluation.")
        return
    indices = random.sample(range(total), 500)
    img_sample = img_embeds[indices]
    txt_sample = txt_embeds[indices]
    captions_sample = [captions_list[i] for i in indices]
    
    # Convert global anchors to tensors.
    anchors_img_tensor = torch.tensor(global_anchors_img, device=img_sample.device, dtype=img_sample.dtype)
    anchors_txt_tensor = torch.tensor(global_anchors_txt, device=txt_sample.device, dtype=txt_sample.dtype)
    
    # --- Compute relative representations ---
    if dist_metric == "mahalanobis":
        img_rel = relative_projection_mahalanobis_batched(img_sample, anchors_img_tensor, global_inv_cov_img)
        txt_rel = relative_projection_mahalanobis_batched(txt_sample, anchors_txt_tensor, global_inv_cov_txt)
    elif dist_metric == "cosine":
        img_rel = relative_projection_cosine(img_sample, anchors_img_tensor)
        txt_rel = relative_projection_cosine(txt_sample, anchors_txt_tensor)
    else:
        raise ValueError("Invalid dist_metric. Choose 'mahalanobis' or 'cosine'.")
    
    # --- Classification using Euclidean distances ---
    dists = torch.cdist(img_rel, txt_rel, p=2)
    print(f"\nEuclidean distance range: min {dists.min().item():.4f}, max {dists.max().item():.4f}")
    
    targets = torch.arange(img_rel.size(0), device=img_rel.device)
    top1_pred = dists.argmin(dim=1)
    top1_correct = (top1_pred == targets).sum().item()
    sorted_dists, sorted_indices = torch.sort(dists, dim=1)
    top5_indices = sorted_indices[:, :5]
    top5_correct = sum(1 for i in range(dists.size(0)) if targets[i] in top5_indices[i])
    
    top1_acc = 100 * top1_correct / dists.size(0)
    top5_acc = 100 * top5_correct / dists.size(0)
    
    print("\nValidation Results on 500 random samples (using Euclidean distances):")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    
    for i in range(min(3, dists.size(0))):
        sorted_vals, sorted_idxs = torch.sort(dists[i], dim=0)
        print(f"\nImage {i} top-5 predictions (lower distance is better):")
        for rank, (val, idx) in enumerate(zip(sorted_vals[:5], sorted_idxs[:5])):
            print(f"  Rank {rank+1}: Caption: \"{captions_sample[idx]}\"   Distance: {val.item():.4f}")
        source_distance = dists[i, i].item()
        print(f"Source Caption: \"{captions_sample[i]}\"   Distance: {source_distance:.4f}")
    
    # --- Plot latent spaces and relative spaces using PCA ---
    if plot_spaces:
        # Use the same 500-sample as above. Reduce each to 2D.
        pca_img = PCA(n_components=2)
        pca_txt = PCA(n_components=2)
        pca_rel_img = PCA(n_components=2)
        pca_rel_txt = PCA(n_components=2)
        
        img_latent_2d = pca_img.fit_transform(img_sample.numpy())
        txt_latent_2d = pca_txt.fit_transform(txt_sample.numpy())
        img_rel_2d = pca_rel_img.fit_transform(img_rel.numpy())
        txt_rel_2d = pca_rel_txt.fit_transform(txt_rel.numpy())
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0,0].scatter(img_latent_2d[:,0], img_latent_2d[:,1], c='blue', alpha=0.6)
        axs[0,0].set_title("Original Image Embeddings")
        axs[0,1].scatter(txt_latent_2d[:,0], txt_latent_2d[:,1], c='green', alpha=0.6)
        axs[0,1].set_title("Original Text Embeddings")
        axs[1,0].scatter(img_rel_2d[:,0], img_rel_2d[:,1], c='red', alpha=0.6)
        axs[1,0].set_title("Relative Image Representations")
        axs[1,1].scatter(txt_rel_2d[:,0], txt_rel_2d[:,1], c='orange', alpha=0.6)
        axs[1,1].set_title("Relative Text Representations")
        for ax in axs.flat:
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        plt.tight_layout()
        # Save the plot to the current folder instead of displaying it.
        plot_path = os.path.join(embeddings_dir, "pca_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"PCA plot saved to: {plot_path}")
    
if __name__ == "__main__":
    main()