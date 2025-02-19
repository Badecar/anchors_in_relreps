import numpy as np
import random
from scipy.spatial.distance import pdist
from tqdm import tqdm
import torch

def objective_function(embeddings, anchors, Coverage_weight=1, diversity_weight=1, exponent=0.5):
    def diversity(embeddings, anchors):
        return (1/len(embeddings)) * sum([min([
                                        abs(pdist([embedding, anchor], metric="cosine")) for anchor in anchors])
                                        for embedding in embeddings])
    
    def coverage(anchors, exponent):
        dists = pdist(anchors, metric="cosine")
        return sum(abs(dist)**exponent for dist in dists)
    
    return (diversity_weight * diversity(embeddings, anchors) - Coverage_weight * coverage(anchors, exponent))[0]

def greedy_one_at_a_time(embeddings, indices, num_anchors, Coverage_weight=1, diversity_weight=3, exponent=1):
    """
    Select anchors greedily by maximizing a trade-off between diversity and coverage.
    
    Parameters:
      embeddings: list or numpy array of shape (n, d)
      indices: array-like indices corresponding to embeddings
      num_anchors: number of anchors to select
      Coverage_weight: weight for the coverage term (to subtract)
      diversity_weight: weight for the diversity term (to add)
      exponent: exponent used in the coverage calculation
      
    Returns:
      anchors_idx: list of selected indices.
    """
    embeddings = np.array(embeddings)  # (n, d)
    indices = np.array(indices)

    # Normalize embeddings once for cosine computations.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Randomly select the first anchor.
    anchors_idx = []
    init_idx = random.sample(list(indices), 1)[0]
    anchors_idx.append(init_idx)

    # Compute initial cosine distances from all embeddings to the first anchor.
    # Cosine distance = 1 - cosine similarity.
    chosen_anchor = normalized_embeddings[init_idx]  # shape (d,)
    # Dot product: shape (n,)
    min_dists = 1 - np.dot(normalized_embeddings, chosen_anchor)

    # Function to compute coverage using pdist on the anchor set.
    def compute_coverage(anchor_array):
        # anchor_array should have shape (m, d) where m is small.
        # pdist returns pairwise distances; raise each to the given exponent and sum.
        if len(anchor_array) <= 1:
            return 0
        return np.sum(abs(pdist(anchor_array, metric="cosine")) ** exponent)

    # Greedily add anchors.
    for _ in tqdm(range(num_anchors - 1), desc="Selecting anchors"):
        best_index = None
        best_score = -np.inf
        best_new_min_dists = None

        # Evaluate each candidate index that is not already selected.
        for candidate in indices:
            if candidate in anchors_idx:
                continue

            # Get candidate's normalized vector.
            candidate_vec = normalized_embeddings[candidate]
            # Compute cosine distances from all embeddings to this candidate in a vectorized way.
            candidate_dists = np.dot(normalized_embeddings, candidate_vec)

            # New minimum distances if candidate were added.
            new_min_dists = np.minimum(np.abs(min_dists), np.abs(candidate_dists))
            diversity_val = np.mean(new_min_dists)  # diversity is the average min distance.

            # Compute coverage for anchors + candidate.
            current_anchor_vectors = normalized_embeddings[anchors_idx]  # shape (m, d)
            candidate_anchor_array = np.vstack([current_anchor_vectors, candidate_vec])
            coverage_val = compute_coverage(candidate_anchor_array)

            # Overall objective: maximize diversity while penalizing coverage.
            current_score = -diversity_weight * diversity_val + Coverage_weight * coverage_val

            if current_score > best_score:
                best_score = current_score
                best_index = candidate
                best_new_min_dists = new_min_dists

        # Update the selected anchors and the min_dists.
        anchors_idx.append(best_index)
        min_dists = best_new_min_dists

    return anchors_idx



### HYPERTUNING ###
# # Define a grid of hyperparameters for anchor selection
# coverage_weights = [0.5, 1, 2, 4]
# diversity_weights = [0.5, 1, 2, 5]
# exponents = [0.25, 0.5, 0.75, 1, 1.5]

# results = []

# total_configurations = len(coverage_weights) * len(diversity_weights) * len(exponents)
# for cov_w, div_w, exp_val in tqdm.tqdm(product(coverage_weights, diversity_weights, exponents), total=total_configurations, desc="Hyperparameter Search"):
#     print(f"\nTrying configuration: Coverage_weight={cov_w}, diversity_weight={div_w}, exponent={exp_val}")
#     # Select anchors using the current hyperparameters
#     greedy_anchor_ids = greedy_one_at_a_time(
#         embeddings_list[0],
#         indices_list[0],
#         anchor_num,
#         Coverage_weight=cov_w,
#         diversity_weight=div_w,
#         exponent=exp_val
#     )
#     # Get anchor embeddings
#     anchors_list_current = select_anchors_by_id(
#         AE_list,
#         embeddings_list,
#         indices_list,
#         greedy_anchor_ids,
#         test_loader.dataset,
#         show=False,
#         device=device
#     )
#     # Compute the relative coordinates based on these anchors
#     relative_coords_list_current = compute_relative_coordinates(
#         embeddings_list,
#         anchors_list_current,
#         flatten=False
#     )
#     # Evaluate the resulting latent space similarity measures
#     mrr_matrix, mean_mrr, cos_sim_matrix, mean_cos_sim = compare_latent_spaces(
#         relative_coords_list_current,
#         indices_list,
#         compute_mrr=compute_mrr,
#         AE_list=AE_list
#     )
#     print(f"Mean MRR: {mean_mrr:.4f}  |  Mean Cosine Similarity: {mean_cos_sim:.4f}")
#     results.append({
#         "Coverage_weight": cov_w,
#         "diversity_weight": div_w,
#         "exponent": exp_val,
#         "mean_mrr": mean_mrr,
#         "mean_cos_sim": mean_cos_sim
#     })
# # Print all results
# print("\nAll configurations and their results:")
# for result in results:
#     print(result)


# best_config_cos = max(results, key=lambda x: x["mean_cos_sim"])
# print("\nBest configuration based on Mean cos sim:")
# print(best_config_cos)

# best_config_mrr = max(results, key=lambda x: x["mean_mrr"])
# print("\nBest configuration based on Mean MRR:")
# print(best_config_mrr)
