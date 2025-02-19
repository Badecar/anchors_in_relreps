import numpy as np
import random
from scipy.spatial.distance import pdist
from tqdm import tqdm
import torch

def objective_function(embeddings, anchors, Coverage_weight=1, diversity_weight=1, exponent=0.5):
    def coverage(embeddings, anchors):
        return (1/len(embeddings)) * sum([min([
                                        abs(pdist([embedding, anchor], metric="euclidean")) for anchor in anchors])
                                        for embedding in embeddings])
    
    def diversity(anchors, exponent):
        dists = pdist(anchors, metric="euclidean")
        return sum(abs(dist)**exponent for dist in dists)/len(anchors)
    
    return (diversity_weight * diversity(embeddings, anchors) - Coverage_weight * coverage(anchors, exponent))[0]

def greedy_one_at_a_time(embeddings, indices, num_anchors, Coverage_weight=1, diversity_weight=1, exponent=0.5):
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
    indices = np.array(indices[0])

    # Normalize embeddings once for cosine computations.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    anchors_idx = []

    # Function to compute coverage using pdist on the anchor set.
    def compute_coverage(anchor_array):
        # anchor_array should have shape (m, d) where m is small.
        # pdist returns pairwise distances; raise each to the given exponent and sum.
        if len(anchor_array) <= 1:
            return 0
        return np.sum(pdist(anchor_array, metric="cosine") ** exponent)

    # Greedily add anchors.
    for _ in range(num_anchors - 1):
        best_index = None
        best_score = -np.inf
        best_new_min_dists = None

        # Evaluate each candidate index that is not already selected.
        for candidates in indices:
            if candidates in anchors_idx:
                continue

            # Get candidate's normalized vector.
            candidate_vecs = normalized_embeddings[:,candidates,:]
            # Compute cosine distances from all embeddings to this candidate in a vectorized way.
            candidate_dists_list = [np.dot(normalized_embedding, candidate_vec) for normalized_embedding, candidate_vec in zip(normalized_embeddings, candidate_vecs)]

            # New minimum distances if candidate were added.
            if len(anchors_idx) > 0:
                new_min_dists = [np.minimum(np.abs(min_dists), np.abs(candidate_dists)) for candidate_dists in candidate_dists_list]
                # Compute coverage for anchors + candidate.
                current_anchor_vectors = normalized_embeddings[:,np.array(anchors_idx),:]  # shape (m, d)
                candidate_anchor_array = np.vstack([current_anchor_vectors, candidate_vecs])
            else:
                new_min_dists = np.abs(candidate_dists_list)
                candidate_anchor_array = candidate_vecs
            diversity_val = np.mean(new_min_dists, axis=0)  # diversity is the average min distance.
            coverage_val = compute_coverage(candidate_anchor_array)

            # Overall objective: maximize diversity while penalizing coverage.
            current_score = np.mean(-diversity_weight * diversity_val + Coverage_weight * coverage_val)

            if current_score > best_score:
                best_score = current_score
                best_index = candidates
                best_new_min_dists = new_min_dists

        # Update the selected anchors and the min_dists.
        anchors_idx.append(best_index)
        min_dists = best_new_min_dists

    return anchors_idx

def greedy_one_at_a_time_single(embeddings, indices, num_anchors, Coverage_weight=1, diversity_weight=1, exponent=0.5):
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
    def compute_diversity(anchor_array):
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
        for candidate_vec, idx in zip(embeddings, indices):
            if idx in anchors_idx:
                continue

            # Get candidate's normalized vector.
            # Compute cosine distances from all embeddings to this candidate in a vectorized way.
            candidate_dists = np.dot(normalized_embeddings, candidate_vec)

            # New minimum distances if candidate were added.
            new_min_dists = np.minimum(np.abs(min_dists), np.abs(candidate_dists))
            coverage_val = np.mean(new_min_dists)  # diversity is the average min distance.

            # Compute coverage for anchors + candidate.
            current_anchor_vectors = normalized_embeddings[anchors_idx]  # shape (m, d)
            candidate_anchor_array = np.vstack([current_anchor_vectors, candidate_vec])
            diversity_val = compute_diversity(candidate_anchor_array)

            # Overall objective: maximize diversity while penalizing coverage.
            current_score = -diversity_weight * diversity_val + Coverage_weight * coverage_val

            if current_score > best_score:
                best_score = current_score
                best_index = idx
                best_new_min_dists = new_min_dists

        # Update the selected anchors and the min_dists.
        anchors_idx.append(best_index)
        min_dists = best_new_min_dists

    return anchors_idx

import numpy as np
import random
from scipy.spatial.distance import pdist
from tqdm import tqdm

def greedy_one_at_a_time_single_euclidean(embeddings, indices, num_anchors, Coverage_weight=1, diversity_weight=1, exponent=1, repetitions=1):
    """
    Select anchors greedily by maximizing a trade-off between diversity and coverage,
    using Euclidean distances.
    
    Definitions:
      - Coverage: average Euclidean distance from a data point to its nearest anchor.
      - Diversity: sum of (Euclidean) distances raised to the given exponent between all pairs of anchors.
      
    Parameters:
      embeddings: list or numpy array of shape (n, d)
      indices: array-like indices corresponding to embeddings
      num_anchors: number of anchors to select
      Coverage_weight: weight for the coverage term (points-to-anchor distances)
      diversity_weight: weight for the diversity term (anchor-to-anchor distances)
      exponent: exponent used in the diversity calculation
      
    Returns:
      anchors_idx: list of selected indices.
    """
    embeddings = np.array(embeddings)  # shape (n, d)
    indices = np.array(indices)
    
    # Randomly select the first anchor.
    anchors_idx = []
    init_idx = random.sample(list(indices), 1)[0]
    anchors_idx.append(init_idx)

    # Compute initial Euclidean distances from all embeddings to the first anchor.
    chosen_anchor = embeddings[init_idx]  # shape (d,)


    ### FOR TESTING
    # best = -np.inf
    # second = None
    # for idx in indices:
    #     dist = np.sum(pdist([embeddings[idx], chosen_anchor], metric="euclidean"))
    #     if dist > best:
    #         best = dist
    #         second = idx
    #     if idx == 51623:
    #         print(embeddings[idx])
    # print(second)
    # print(embeddings[second], embeddings[48861], embeddings[anchors_idx[0]], np.sum(pdist([embeddings[second], embeddings[anchors_idx[0]]], metric="euclidean")), np.sum(pdist([embeddings[48861], embeddings[anchors_idx[0]]], metric="euclidean")))
    # anchors_idx.append(second)
    # return anchors_idx
    min_dists = np.linalg.norm(embeddings - chosen_anchor, axis=1)  # shape (n,)
    all_ids = np.array(indices)
    def compute_diversity(anchor_array, exponent=1):
        """
        Compute diversity as the sum of pairwise Euclidean distances (raised to the exponent)
        among the anchors.
        """
        if len(anchor_array) <= 1:
            return 0
        pairwise_distances = pdist(anchor_array, metric="euclidean") ** exponent
        n_pairs = len(pairwise_distances)
        return np.mean(pairwise_distances) / n_pairs
    # Greedily add anchors.
    for _ in tqdm(range(num_anchors*repetitions - 1), desc="Selecting anchors"):
        best_index = None
        best_score = -np.inf
        best_new_min_dists = None
        
        # Evaluate each candidate index not already selected.
        for candidate_vec, idx in zip(embeddings, indices):
            if idx in anchors_idx:
                continue
            
            # Compute Euclidean distances from all embeddings to this candidate.
            candidate_dists = np.linalg.norm(embeddings - candidate_vec, axis=1)
            
            # New minimum distances if candidate were added.
            new_min_dists = np.minimum(min_dists, candidate_dists)
            # "Coverage" is defined as the average point-to-anchor distance.
            coverage_val = np.mean(new_min_dists)
            
            # Compute diversity for the set of anchors if we added this candidate.
            anchors_idx_np = []
            for uid in anchors_idx:
                # Find the position where the dataset id matches the desired id.
                idx_temp = np.where(all_ids == uid)[0]
                if idx_temp.size == 0:
                    raise ValueError(f"ID {uid} not found in the obtained indices.")
                anchors_idx_np.append(idx_temp[0])
            anchors_idx_np = np.array(anchors_idx_np)
            current_anchor_vectors = embeddings[anchors_idx_np]  # current anchors (m, d)
            candidate_anchor_array = np.vstack([current_anchor_vectors, candidate_vec])
            diversity_val = compute_diversity(candidate_anchor_array, exponent=exponent)
            
            # Overall objective: here we subtract diversity and add coverage.
            # (Adjust signs if your formulation is different.)
            current_score = diversity_weight * diversity_val - Coverage_weight * coverage_val
            
            if current_score > best_score:
                best_score = current_score
                best_index = idx
                best_new_min_dists = new_min_dists
                temp = (coverage_val, diversity_val)
        
        # Update the selected anchors and the min_dists.
        min_dists = best_new_min_dists
        print(temp)
        anchors_idx.append(best_index)
        if len(anchors_idx) >= num_anchors and (num_anchors*repetitions - 2) != _:
            del anchors_idx[0]
            dists = np.array([np.linalg.norm(embeddings - embeddings[anchor], axis=1) for anchor in anchors_idx]).T
            min_dists = np.min(dists, axis=0)
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
