import numpy as np
import random
from scipy.spatial.distance import pdist
from tqdm import tqdm

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

def greedy_one_at_a_time_single_cossim(embeddings, indices, num_anchors, Coverage_weight=1, diversity_weight=1, exponent=0.5):
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

def greedy_one_at_a_time_single_euclidean(embeddings_list, indices_list, num_anchors, Coverage_weight=1, diversity_weight=1, exponent=1, repetitions=1, verbose=True):
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
    indices_list = [np.array(indices) for indices in indices_list]
    indices = indices_list[0]
    
    # Randomly select the first anchor.
    anchors_idx = []

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
    for _ in tqdm(range(num_anchors*repetitions), desc="Selecting anchors", disable=not(verbose)):
        if _ == 0:
            init_idx = random.sample(list(indices_list[0]), 1)[0]
            init_idx_multiple = [np.where(indices_temp == init_idx)[0] for indices_temp in indices_list]
            anchors_idx.append(init_idx)

            # Compute initial Euclidean distances from all embeddings to the first anchor.
            chosen_anchors = [embeddings[idx] for embeddings, idx in zip(embeddings_list, init_idx_multiple)]  # shape (d,)
            min_dists_list = [np.linalg.norm(embeddings - chosen_anchor, axis=1) for embeddings, chosen_anchor in zip(embeddings_list, chosen_anchors)]  # shape (n,)
            
            all_ids = np.array(indices)
            continue

        best_index = None
        best_score = -np.inf
        best_new_min_dists = None
        # Evaluate each candidate index not already selected.
        """
        only loop through the indices to simplify the loop
        """
        for idx in tqdm(indices, desc="checking indices", disable=True):
            if idx in anchors_idx:
                continue
            
            """
            make a list of the vectors with a given index
            """
            # TODO
            candidate_vec_list = []

            for embedding, ids in zip(embeddings_list, indices_list):
                candidate_vec_list.append(np.array(embedding[ids == idx][0]))
            # Compute Euclidean distances from all embeddings to this candidate.
            """
            make candidate_dists into a list, with a element for each embedding sapce
            """
            candidate_dists_list = [np.linalg.norm(embeddings - candidate_vec, axis=1) for embeddings, candidate_vec in zip(embeddings_list, candidate_vec_list)]
            # New minimum distances if candidate were added.
            """
            make new_min_dists into a list of the new minium distances in each embedding space
            """
            new_min_dists_list = [np.minimum(min_dists, candidate_dists) for min_dists, candidate_dists in zip(min_dists_list, candidate_dists_list)]
            # "Coverage" is defined as the average point-to-anchor distance.

            coverage_val = np.mean(np.mean(np.array(new_min_dists_list), axis=1)**2)
            
            # Compute diversity for the set of anchors if we added this candidate.

            """
            make all this into a loop to get a list of the indecies in the diffrent embedding spaces
            ideally the indecies should be the same if the embeddings are sorted with releation to thier index, but in case they are not
            """
            diversity_val = []
            # loop should start here
            for embeddings, candidate_vecs in zip(embeddings_list, candidate_vec_list):
                anchors_idx_np = []
                for uid in anchors_idx:
                    # Find the position where the dataset id matches the desired id.
                    idx_temp = np.where(all_ids == uid)[0]
                    if idx_temp.size == 0:
                        raise ValueError(f"ID {uid} not found in the obtained indices.")
                    anchors_idx_np.append(idx_temp[0])
                
                anchors_idx_np = np.array(anchors_idx_np)
                current_anchor_vectors = embeddings[anchors_idx_np]  # current anchors (m, d)
                candidate_anchor_array = np.vstack([current_anchor_vectors, candidate_vecs])
                diversity_val.append(compute_diversity(candidate_anchor_array, exponent=exponent))
            diversity_val = np.mean(np.array(diversity_val))
            # loop should end here
            
            # Overall objective: here we subtract diversity and add coverage.
            # (Adjust signs if your formulation is different.)
            current_score = diversity_weight * diversity_val - Coverage_weight * coverage_val

            if current_score > best_score:
                best_score = current_score
                best_index = idx
                best_new_min_dists = new_min_dists_list
        
        # Update the selected anchors and the min_dists.
        min_dists_list = best_new_min_dists
        anchors_idx.append(best_index)
        if len(anchors_idx) >= num_anchors and (num_anchors*repetitions - 1) != _:
            del anchors_idx[0]

            """
            dists and min dists should be for each embedding space
            """
            dists_list = [np.array([np.linalg.norm(embeddings - embeddings[indices==index], axis=1) for anchor, index in zip(anchors_idx, indices)]).T for embeddings, indices in zip(embeddings_list, indices_list)]
            min_dists_list = [np.min(dists, axis=1) for dists in dists_list]

    return anchors_idx
