import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv(metric, decoder="rexnet_100", encoder="vit_small_patch16_224"):
    current_path = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_path, f"results_{decoder}_enc_{encoder}_{metric}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    return pd.read_csv(csv_file)

def plot_results(ax, df, metric):
    anchors = df["Anchors"].values
    methods = [
        ("Random", "Random_mean", "Random_std"),
        ("KMeans", "KMeans_mean", "KMeans_std")
    ]
    for label, mean_col, std_col in methods:
        means = df[mean_col].values
        stds = df[std_col].values
        ax.plot(anchors, means, marker="o", label=label)
        ax.fill_between(anchors, means - stds, means + stds, alpha=0.2)
    ax.set_xlabel("Number of Anchors")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title(f"Zero-Shot Classification F1 Using {metric}")
    ax.legend()
    ax.grid(True)

def main():
    metrics = ["cosine", "mahalanobis", "euclidean"]
    csv_dfs = {}
    
    # Load all CSV data
    for metric in metrics:
        csv_dfs[metric] = load_csv(metric)
    
    # Compute common x limits across all CSVs
    all_anchors = np.concatenate([df["Anchors"].values for df in csv_dfs.values()])
    x_min, x_max = np.min(all_anchors), np.max(all_anchors)
    
    # Compute common y limits across all methods and CSVs
    y_mins = []
    y_maxs = []
    methods = [
        ("Random", "Random_mean", "Random_std"),
        ("KMeans", "KMeans_mean", "KMeans_std")
    ]
    for df in csv_dfs.values():
        for label, mean_col, std_col in methods:
            means = df[mean_col].values
            stds = df[std_col].values
            y_mins.append(np.min(means - stds))
            y_maxs.append(np.max(means + stds))
    y_min, y_max = np.min(y_mins)-4, np.max(y_maxs)+4
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=True)
    for idx, metric in enumerate(metrics):
        df = csv_dfs[metric]
        plot_results(axes[idx], df, metric)
        axes[idx].set_xlim(x_min, x_max)
        axes[idx].set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()