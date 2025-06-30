import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..', '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import numpy as np
import torch
import matplotlib.pyplot as plt
import csv

from relreps import get_relrep
from models import build_classifier, train_classifier, evaluate_classifier
from utils import set_random_seeds
from anchors.P_anchors import get_P_anchors
from data import get_features_dict
from anchors import get_kmeans_anchors

def run_experiment(dist_metric, dataset_name="cifar100", plot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_perc = 1.0
    batch_size = 64
    num_epochs = 9
    n_closest = 100
    n_seeds = 7

    anchor_nums = list(range(200, 800, 50))

    transformers = [
        "vit_base_patch16_224",
        "rexnet_100",
        "vit_base_resnet50_384",
        "vit_small_patch16_224"
    ]

    baseline_transformer = "vit_base_patch16_224"
    zeroshot_transformer = "rexnet_100"

    print(f"Distance metric for this run: {dist_metric}")

    features_dict, num_classes = get_features_dict(dataset_name, train_perc, batch_size, device, transformers)

    # Select features from each transformer.
    enc1_feats_train = features_dict[baseline_transformer]["train_features"]
    enc1_labels_train = features_dict[baseline_transformer]["train_labels"]
    enc1_feats_test = features_dict[baseline_transformer]["test_features"]
    enc1_labels_test = features_dict[baseline_transformer]["test_labels"]

    enc2_feats_train = features_dict[zeroshot_transformer]["train_features"]
    enc2_feats_test = features_dict[zeroshot_transformer]["test_features"]
    enc2_labels_test = features_dict[zeroshot_transformer]["test_labels"]

    emb_list = [enc1_feats_test.cpu().numpy(), enc2_feats_test.cpu().numpy()]

    results_random = {a: [] for a in anchor_nums}
    results_optimized = {a: [] for a in anchor_nums}
    results_kmeans = {a: [] for a in anchor_nums}
    
    for num_anchors in anchor_nums:
        print(f"\nEvaluating for {num_anchors} anchors...")
        for seed in range(42, 42 + n_seeds):
            print(f" Seed {seed}")
            set_random_seeds(seed)
            sample_count = enc1_feats_train.shape[0]
            random_indices = np.sort(np.random.choice(sample_count, num_anchors, replace=False))

            # --- RANDOM ANCHORS ---
            random_anch_enc1 = enc1_feats_train[random_indices].to(device)
            random_anch_enc2 = enc2_feats_train[random_indices].to(device)
            relrep_train_random = get_relrep(enc1_feats_train.to(device), random_anch_enc1, dist_metric, device)
            relrep_test_random = get_relrep(enc2_feats_test.to(device), random_anch_enc2, dist_metric, device)

            # --- P OPTIMIZED ANCHORS ---
            optimize_distmeasure = "euclidean" if dist_metric != "cosine" else "cosine" # use eucl for mahalanobis
            _, P_anchors, _ = get_P_anchors(
                emb=emb_list,
                anchor_num=num_anchors,
                clustered=True, # If True uses the anchor coordinates found by P and redefines them as a comb of the n_closests
                n_closest=n_closest,
                epochs=400,
                lr=1e-2,
                coverage_weight=1, # Diversity_w = 1-cov_w
                anti_collapse_w=0,
                exponent=1,
                dist_measure=optimize_distmeasure,
                verbose=False,
                device=device
            )
            P_anch_enc1, P_anch_enc2 = P_anchors[0], P_anchors[1]
            relrep_train_optimized = get_relrep(enc1_feats_train.to(device), P_anch_enc1, dist_metric, device)
            relrep_test_optimized = get_relrep(enc2_feats_test.to(device), P_anch_enc2, dist_metric, device)

            # --- KMEANS ANCHORS ---
            print("Computing KMeans-based datapoint anchors on enc1")
            anch_list_kmeans, _ = get_kmeans_anchors(
                embeddings=emb_list,
                anchor_num=num_anchors,
                n_closest=n_closest,
                kmeans_seed=seed,
                verbose=False
            )
            kmeans_anch_enc1_np, kmeans_anch_enc2_np = anch_list_kmeans[0], anch_list_kmeans[1]
            kmeans_anch_enc1 = torch.tensor(kmeans_anch_enc1_np, device=device, dtype=enc1_feats_test.dtype)
            kmeans_anch_enc2 = torch.tensor(kmeans_anch_enc2_np, device=device, dtype=enc2_feats_test.dtype)
            relrep_train_kmeans = get_relrep(enc1_feats_train.to(device), kmeans_anch_enc1, dist_metric, device)
            relrep_test_kmeans = get_relrep(enc2_feats_test.to(device), kmeans_anch_enc2, dist_metric, device)


            # --- TRAINING RELATIVE CLASSIFIERS ---
            # Building classifiers
            clf_random = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            clf_optimized = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            clf_kmeans = build_classifier(num_anchors, num_anchors, num_classes).to(device)
            
            # Training classifiers
            clf_random = train_classifier(clf_random, relrep_train_random, enc1_labels_train.to(device), device, num_epochs, type="random")
            clf_optimized = train_classifier(clf_optimized, relrep_train_optimized, enc1_labels_train.to(device), device, num_epochs, type="P_optimized")
            clf_kmeans = train_classifier(clf_kmeans, relrep_train_kmeans, enc1_labels_train.to(device), device, num_epochs, type="kmeans")

            # Evaluating zero shot classification
            f1_random = evaluate_classifier(clf_random, relrep_test_random, enc2_labels_test.to(device), device)
            f1_optimized = evaluate_classifier(clf_optimized, relrep_test_optimized, enc2_labels_test.to(device), device)
            f1_kmeans = evaluate_classifier(clf_kmeans, relrep_test_kmeans, enc2_labels_test.to(device), device)
            print(f"Dist: {dist_metric} --  Random F1: {f1_random:.2f}%, Optimized F1: {f1_optimized:.2f}%, KMeans F1: {f1_kmeans:.2f}%")

            results_random[num_anchors].append(f1_random)
            results_optimized[num_anchors].append(f1_optimized)
            results_kmeans[num_anchors].append(f1_kmeans)


    # --- Save results to CSV ---
    anchor_nums_arr = []
    random_means, random_stds = [], []
    optimized_means, optimized_stds = [], []
    kmeans_means, kmeans_stds = [], []
    for a in anchor_nums:
        anchor_nums_arr.append(a)
        r_mean = np.mean(results_random[a])
        r_std = np.std(results_random[a])
        o_mean = np.mean(results_optimized[a])
        o_std = np.std(results_optimized[a])
        k_mean = np.mean(results_kmeans[a])
        k_std = np.std(results_kmeans[a])
        random_means.append(r_mean)
        random_stds.append(r_std)
        optimized_means.append(o_mean)
        optimized_stds.append(o_std)
        kmeans_means.append(k_mean)
        kmeans_stds.append(k_std)
        print(f"Anchors: {a}, Random: {r_mean:.2f}% ± {r_std:.2f}%, Optimized: {o_mean:.2f}% ± {o_std:.2f}%, KMeans: {k_mean:.2f}% ± {k_std:.2f}%")

    csv_file = os.path.join(current_path, f"results_{baseline_transformer}_enc_{zeroshot_transformer}_{dist_metric}.csv")
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Anchors", "Random_mean", "Random_std", "Optimized_mean", "Optimized_std", "KMeans_mean", "KMeans_std"])
        for a, r_mean, r_std, o_mean, o_std, k_mean, k_std in zip(anchor_nums_arr, random_means, random_stds, optimized_means, optimized_stds, kmeans_means, kmeans_stds):
            writer.writerow([a, r_mean, r_std, o_mean, o_std, k_mean, k_std])
    print(f"Results saved to {csv_file}")
    
    # Free memory and clear GPU cache if needed.
    del features_dict, enc1_feats_train, enc1_feats_test, enc2_feats_train, enc2_feats_test
    torch.cuda.empty_cache()

    if plot:
        plt.figure()
        plt.plot(anchor_nums_arr, random_means, label="Random", marker='o')
        plt.fill_between(anchor_nums_arr,
                        np.array(random_means) - np.array(random_stds),
                        np.array(random_means) + np.array(random_stds),
                        alpha=0.2)
        plt.plot(anchor_nums_arr, optimized_means, label="Optimized", marker='o')
        plt.fill_between(anchor_nums_arr,
                        np.array(optimized_means) - np.array(optimized_stds),
                        np.array(optimized_means) + np.array(optimized_stds),
                        alpha=0.2)
        plt.plot(anchor_nums_arr, kmeans_means, label="KMeans", marker='o')
        plt.fill_between(anchor_nums_arr,
                        np.array(kmeans_means) - np.array(kmeans_stds),
                        np.array(kmeans_means) + np.array(kmeans_stds),
                        alpha=0.2)
        plt.xlabel("Number of Anchors")
        plt.ylabel("F1 Score (%)")
        plt.title(f"Relative Classifier Performance vs Number of Anchors ({dist_metric})")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # To run on ImageNet1k, call run_experiment with dataset_name="imagenet1k"
    # Beware: ImageNet is huge
    for metric in ["mahalanobis", "cosine", "euclidean"]:
        print("\n======================================")
        print(f"Starting run with distance metric: {metric}")
        print("======================================\n")
        run_experiment(metric, dataset_name="CIFAR100_coarse", plot=False)
        
