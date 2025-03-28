import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from itertools import product
from tqdm import tqdm

# -------------------------------
# Dataset Creation Functions (Works for arbitrary dimensions)
# -------------------------------
def create_cluster(center, cov, n_points, label):
    """Generates n_points for a single cluster with given center and covariance."""
    points = np.random.multivariate_normal(mean=center, cov=cov, size=n_points)
    labels = np.full(n_points, label)
    return points, labels

def create_dataset(centers, cov_list, points_per_cluster):
    """
    Creates a synthetic dataset.
    centers: list of arrays of shape [D]
    cov_list: list of covariance matrices of shape [D, D] for each cluster.
    points_per_cluster: number of points per cluster.
    Returns:
      X: np.array of shape [N, D]
      y: np.array of labels of shape [N,]
    """
    data, labels = [], []
    for i, center in enumerate(centers):
        cov = cov_list[i]
        pts, lab = create_cluster(center, cov, points_per_cluster, i)
        data.append(pts)
        labels.append(lab)
    return np.vstack(data), np.hstack(labels)

# -------------------------------
# Relative Representation Functions in PyTorch
# -------------------------------
def compute_relrep_cosine_torch(X, anchors):
    # X: [N, D], anchors: [num_anc, D]
    X_unit = X / (X.norm(p=2, dim=1, keepdim=True) + 1e-8)
    anchors_unit = anchors / (anchors.norm(p=2, dim=1, keepdim=True) + 1e-8)
    return X_unit @ anchors_unit.t()

def compute_relrep_euclidean_torch(X, anchors):
    diff = X.unsqueeze(1) - anchors.unsqueeze(0)  # [N, num_anc, D]
    return diff.norm(p=2, dim=2)

def compute_relrep_mahalanobis_torch(X, anchors):
    # Compute sample covariance of X (along dim0)
    X_centered = X - X.mean(dim=0, keepdim=True)
    cov = (X_centered.t() @ X_centered) / (X.shape[0]-1)
    # Add a small ridge to ensure invertibility
    cov = cov + 1e-6 * torch.eye(cov.shape[0], device=X.device)
    inv_cov = torch.inverse(cov)
    diff = X.unsqueeze(1) - anchors.unsqueeze(0)  # [N, num_anc, D]
    d2 = torch.einsum('nij,jk,nik->ni', diff, inv_cov, diff)
    return torch.sqrt(d2 + 1e-8)

# -------------------------------
# PyTorch Logistic Regression Model
# -------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def train_logistic_regression(X_train, y_train, X_test, y_test, input_dim, num_classes, device, epochs=200, batch_size=1024, lr=0.1):
    model = LogisticRegressionModel(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        outputs = model(X_test.to(device))
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
    return f1_score(y_test.cpu().numpy(), preds, average='weighted')

# -------------------------------
# Main experimental loop
# -------------------------------
if __name__ == '__main__':
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Choose representative dimensions for the experiment; extend this range as needed.
    Ds = [20, 50, 100, 250, 500, 750, 1000, 1500]
    # Anchors ratio list (num_anc = ratio * D)
    anc_factors = [0.2, 0.5, 0.75, 1.0, 1.2]
    # n_clusters factor (n_clusters = factor * D)
    clust_factors = [1, 2, 3, 4]
    points_per_cluster = 40
    seeds = range(15)
    
    # Build all configurations as products
    configs = list(product(Ds, anc_factors, clust_factors, seeds))
    
    results = []
    pbar = tqdm(configs, total=len(configs), desc="Total Configurations")
    
    for D, anc_factor, clust_factor, seed in pbar:
        num_anc = max(1, int(anc_factor * D))
        n_clusters = max(1, int(clust_factor * D))
        num_classes = n_clusters  # labels range from 0 to n_clusters-1
        
        # Set both numpy and torch seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate cluster centers and covariances
        centers = [np.random.uniform(-10, 10, size=D) for _ in range(n_clusters)]
        cov_list = []
        for _ in range(n_clusters):
            A = np.random.randn(D, D)
            cov = np.dot(A, A.T) + np.eye(D) * 0.1
            cov_list.append(cov)
        
        # Create dataset (using numpy functions)
        X_np, y_np = create_dataset(centers, cov_list, points_per_cluster)
        # Scale data
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X_np)
        # Generate anchors in [0,10]^D and scale them similarly
        anchors_np = [np.random.uniform(0, 10, size=D) for _ in range(num_anc)]
        anchors_np = [scaler.transform(a.reshape(1, -1)).flatten() for a in anchors_np]
        
        # Split data (using train_test_split)
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_np, y_np, test_size=0.3, random_state=42)
        # Convert to torch tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        y_test = torch.tensor(y_test_np, dtype=torch.long)
        
        # Prepare anchors tensor
        anchors_tensor = torch.tensor(np.array(anchors_np), dtype=torch.float32)
        
        # Train on absolute data
        t0 = time.time()
        f1_abs = train_logistic_regression(X_train, y_train, X_test, y_test, D, num_classes, device)
        
        # Compute relative representations on training and test sets on GPU
        X_train_gpu = X_train.to(device)
        X_test_gpu = X_test.to(device)
        anchors_gpu = anchors_tensor.to(device)
        
        rel_cosine_train = compute_relrep_cosine_torch(X_train_gpu, anchors_gpu)
        rel_cosine_test = compute_relrep_cosine_torch(X_test_gpu, anchors_gpu)
        rel_euclidean_train = compute_relrep_euclidean_torch(X_train_gpu, anchors_gpu)
        rel_euclidean_test = compute_relrep_euclidean_torch(X_test_gpu, anchors_gpu)
        rel_mahalanobis_train = compute_relrep_mahalanobis_torch(X_train_gpu, anchors_gpu)
        rel_mahalanobis_test = compute_relrep_mahalanobis_torch(X_test_gpu, anchors_gpu)
        
        # Train logistic regression models (using torch) on each relative representation
        f1_cosine = train_logistic_regression(rel_cosine_train.cpu(), y_train, rel_cosine_test.cpu(), y_test, num_anc, num_classes, device, epochs=200)
        f1_euclidean = train_logistic_regression(rel_euclidean_train.cpu(), y_train, rel_euclidean_test.cpu(), y_test, num_anc, num_classes, device, epochs=200)
        f1_mahalanobis = train_logistic_regression(rel_mahalanobis_train.cpu(), y_train, rel_mahalanobis_test.cpu(), y_test, num_anc, num_classes, device, epochs=200)
        t1 = time.time()
        
        results.append({
            "seed": seed,
            "D": D,
            "num_anc": num_anc,
            "n_clusters": n_clusters,
            "f1_abs": f1_abs,
            "f1_cosine": f1_cosine,
            "f1_euclidean": f1_euclidean,
            "f1_mahalanobis": f1_mahalanobis,
            "runtime_sec": t1 - t0
        })
        pbar.set_description(f"D={D}, anc={num_anc}, clust={n_clusters} | abs={f1_abs:.4f}, cos={f1_cosine:.4f}, euc={f1_euclidean:.4f}, mahal={f1_mahalanobis:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    
    print("Experiment completed. Results saved to experiment_results.csv")