import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def read_csv(file_path):
    n_closest = []
    mean_f1 = []
    std_f1 = []
    with open(file_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            n_closest.append(int(float(row["n_closest"])))
            mean_f1.append(float(row["Mean_F1"]))
            std_f1.append(float(row["Std_F1"]))
    return np.array(n_closest), np.array(mean_f1), np.array(std_f1)

# Update these paths with your actual CSV files.
current_path = os.path.dirname(os.path.abspath(__file__))
csv_file_200 = os.path.join(current_path, "results_nclosestplot_200anch_kmeans_mahalanobis_base_vit_base_resnet50_384_target_vit_small_patch16_224.csv")
csv_file_768 = os.path.join(current_path, "results_nclosestplot_768anch_kmeans_mahalanobis_base_vit_base_resnet50_384_target_vit_small_patch16_224.csv")

n_closest_200, mean_f1_200, std_f1_200 = read_csv(csv_file_200)
n_closest_768, mean_f1_768, std_f1_768 = read_csv(csv_file_768)

fig, ax = plt.subplots(figsize=(10,6))

# Plot for 200 anchors
ax.plot(n_closest_200, mean_f1_200, marker="o", label="200 Anchors")
ax.fill_between(n_closest_200, mean_f1_200 - std_f1_200, mean_f1_200 + std_f1_200, alpha=0.2)

# Plot for 768 anchors
ax.plot(n_closest_768, mean_f1_768, marker="o", label="768 Anchors")
ax.fill_between(n_closest_768, mean_f1_768 - std_f1_768, mean_f1_768 + std_f1_768, alpha=0.2)

ax.set_xlabel("n_closest")
ax.set_ylabel("F1 Score (%)")
ax.set_title("F1 Score vs n_closest for 200 and 768 Anchors")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()