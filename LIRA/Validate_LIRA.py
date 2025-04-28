import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import csv

from sentence_transformers import SentenceTransformer

# Import our SBU dataset and collate function from LIRA3.py
from LIRA3 import SBUDataset, custom_collate_fn

# ----- Relative representation functions (as we always use) -----
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

# ----- Main validation procedure -----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load previously computed anchors (assumes they were saved by LIRA3.py)
    anchors_file = os.path.join(os.path.dirname(__file__), "lira_kmeans_sbu_anchors.pt")
    if not os.path.exists(anchors_file):
        print(f"Anchors file not found at {anchors_file}.")
        return
    checkpoint = torch.load(anchors_file, map_location=device)
    # We use the parallel anchors for each modality.
    anchors_img = checkpoint["anchors_img"].numpy()   # for image relative rep
    anchors_txt = checkpoint["anchors_txt"].numpy()   # for text relative rep

    num_anchors = anchors_img.shape[0]  # number of anchors

    # Load 500 random SBU samples.
    preprocess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    full_dataset = SBUDataset(split="train", transform=preprocess_img, max_samples=10000, csv_filename="sbu_captions.csv")
    total_samples = len(full_dataset)
    sample_indices = random.sample(range(total_samples), 500)
    dataset = Subset(full_dataset, sample_indices)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    # Build the image encoder (ResNet-50 without classification head).
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Replace fc with identity to get embeddings.
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    # Build the text encoder.
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Collect images and captions.
    all_images = []
    all_captions = []
    for batch in tqdm(loader, desc="Loading SBU samples"):
        if batch is None:
            continue
        imgs, caps = batch
        all_images.append(imgs)
        all_captions.extend(caps)
    images_tensor = torch.cat(all_images, dim=0)  # Shape: [500, C, H, W]
    # There should be 500 captions in all_captions.

    # ---------- Embedding extraction ----------
    # Compute text embeddings (for captions).
    print("Extracting text embeddings...")
    with torch.no_grad():
        txt_embeds = text_encoder.encode(all_captions, convert_to_tensor=True).to(device)
    # Compute image embeddings.
    print("Extracting image embeddings...")
    with torch.no_grad():
        img_embeds = resnet(images_tensor.to(device))
    
    # ---------- Compute relative representations ----------
    # For text: use anchors_txt and covariance computed on text embeddings.
    cov_txt = compute_covariance_matrix(txt_embeds)
    inv_cov_txt = torch.linalg.inv(cov_txt + 1e-6 * torch.eye(cov_txt.size(0), device=device))
    # Our standard framework: project embeddings relative to anchors.
    text_rel = relative_projection_mahalanobis_batched(txt_embeds, anchors_txt, inv_cov_txt)
    # For images: use anchors_img
    cov_img = compute_covariance_matrix(img_embeds)
    inv_cov_img = torch.linalg.inv(cov_img + 1e-6 * torch.eye(cov_img.size(0), device=device))
    image_rel = relative_projection_mahalanobis_batched(img_embeds, anchors_img, inv_cov_img)

    # ---------- Classification via nearest-neighbor (softmax over euclidean distances) ----------
    # Compute Euclidean distances between each image relative rep and every text relative rep.
    # We'll use torch.cdist on GPU.
    image_rel = image_rel.to(device)
    text_rel = text_rel.to(device)
    dists = torch.cdist(image_rel, text_rel, p=2)  # Shape: [500, 500]
    # Convert distances to similarity scores (negative distances) and softmax.
    probs = torch.softmax(-dists, dim=1)  # each row is probability distribution over 500 classes

    # Ground truth: sample order is preserved (sample i's caption is the correct class for image i).
    targets = torch.arange(probs.size(0), device=device)
    pred_top1 = probs.argmax(dim=1)
    top1_correct = (pred_top1 == targets).sum().item()

    # Top-5: count an image as correct if true index is among top 5 highest-probability.
    top5_correct = 0
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
    for i in range(probs.size(0)):
        if targets[i] in top5_indices[i]:
            top5_correct += 1
    
    top1_acc = 100 * top1_correct / probs.size(0)
    top5_acc = 100 * top5_correct / probs.size(0)
    
    print(f"\nValidation Results (over 500 samples):")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    # ---------- Show top-5 probability distribution and captions for the first 3 images ----------
    for i in range(3):
        tp, ti = torch.topk(probs[i], k=5)
        print(f"\nImage {i} predictions:")
        for prob, idx in zip(tp.tolist(), ti.tolist()):
            print(f'  Caption: "{all_captions[idx]}"  with probability: {prob:.4f}')

if __name__ == "__main__":
    main()