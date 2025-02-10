import numpy as np
import torch

import torch.nn.functional as F

def compare_latent_spaces(embeddings1, indices1, embeddings2, indices2):
    """
    Computes the Mean Reciprocal Rank (MRR), average cosine similarity, and average Jaccard similarity 
    between two sets of embeddings.
    
    Assumptions:
      - embeddings1 and embeddings2 are Tensors of shape (N, d)
      - indices1 and indices2 are iterables containing the unique dataset IDs for each embedding,
        which allow us to determine the "correct" matching pair (they need not be in the same order).
        
    The function first reorders the embeddings according to the ascending order of their indices.
    
    Args:
        embeddings1 (Tensor): First set of embeddings, shape (N, d).
        indices1 (iterable): Unique dataset IDs for embeddings1.
        embeddings2 (Tensor): Second set of embeddings, shape (N, d).
        indices2 (iterable): Unique dataset IDs for embeddings2.
    
    Returns:
        mrr (float): Mean Reciprocal Rank over all queries.
        mean_cos_sim (float): Mean cosine similarity (elementwise between matching pairs).
        mean_jaccard (float): Mean Jaccard similarity (computed on binarized embeddings).
    """
    # Convert indices to NumPy arrays so we can sort them
    indices1 = np.array(indices1)
    indices2 = np.array(indices2)
    
    # Compute sorting orders based on indices
    order1 = np.argsort(indices1)
    order2 = np.argsort(indices2)
    
    # Reorder embeddings so that they are in the same (sorted) order of indices.
    embeddings1_sorted = embeddings1[order1]
    embeddings2_sorted = embeddings2[order2]
    
    # Calculate the cosine similarity elementwise between corresponding embeddings.
    cos_sim = F.cosine_similarity(embeddings1_sorted, embeddings2_sorted, dim=1)
    mean_cos_sim = cos_sim.mean().item()
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr_total = 0.0
    N = embeddings1_sorted.size(0)
    for i in range(N):
        query = embeddings1_sorted[i].unsqueeze(0)  # shape [1, d]
        # Cosine similarities between the query and all embeddings in embeddings2_sorted.
        sim_all = F.cosine_similarity(query, embeddings2_sorted, dim=1)
        # Get descending order indices.
        _, sorted_indices = torch.sort(sim_all, descending=True)
        # Find rank for the correct match (matching index i) in sorted order (1-indexed).
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        mrr_total += 1.0 / rank
    mrr = mrr_total / N
    
    return mrr, mean_cos_sim