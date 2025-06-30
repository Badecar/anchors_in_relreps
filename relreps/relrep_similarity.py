import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def compute_latent_similarity(embeddings1, indices1, embeddings2, indices2, compute_mrr=False):
    """
    Computes the Mean Reciprocal Rank (MRR), average cosine similarity, and average Jaccard similarity 
    between two sets of embeddings.
    The function first reorders the embeddings according to the ascending order of their indices.
    
    Args:
        embeddings1 (Tensor): First set of embeddings, shape (N, d).
        indices1 (iterable): Unique dataset IDs for embeddings1.
        embeddings2 (Tensor): Second set of embeddings, shape (N, d).
        indices2 (iterable): Unique dataset IDs for embeddings2.
    
    Returns:
        mrr (float): Mean Reciprocal Rank over all queries.
        mean_cos_sim (float): Mean cosine similarity (elementwise between matching pairs).
    """
    
    # Ensure embeddings are torch tensors
    if not torch.is_tensor(embeddings1):
        embeddings1 = torch.from_numpy(embeddings1)
    if not torch.is_tensor(embeddings2):
        embeddings2 = torch.from_numpy(embeddings2)

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
    
    if not compute_mrr:
        return None, mean_cos_sim
    ### FOR LOOP MRR - TIME INTENSIVE ###
    # # Compute Mean Reciprocal Rank (MRR)
    # mrr_total = 0.0
    # N = embeddings1_sorted.size(0)
    # for i in tqdm(range(N), desc='Computing MRR'):
    #     query = embeddings1_sorted[i].unsqueeze(0)  # shape [1, d]
    #     # Cosine similarities between the query and all embeddings in embeddings2_sorted.
    #     sim_all = F.cosine_similarity(query, embeddings2_sorted, dim=1)
    #     # Get descending order indices.
    #     _, sorted_indices = torch.sort(sim_all, descending=True)
    #     # Find rank for the correct match (matching index i) in sorted order (1-indexed).
    #     rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
    #     mrr_total += 1.0 / rank
    # mrr = mrr_total / N

    ### VECTORIZE MRR - MEMORY INTENSIVE ###
    # Normalize embeddings to compute cosine similarities via matrix multiplication.
    embeddings1_norm = embeddings1_sorted / embeddings1_sorted.norm(dim=1, keepdim=True)
    embeddings2_norm = embeddings2_sorted / embeddings2_sorted.norm(dim=1, keepdim=True)
    sim_matrix = torch.mm(embeddings1_norm, embeddings2_norm.t())  # shape (N, N)
    # Mean cosine similarity between corresponding embeddings (diagonal elements)
    mean_cos_sim = sim_matrix.diag().mean().item()
    # Compute the rank for each query: count how many items in each row have a higher similarity than the true match.
    # For each row i, rank = (# of entries > sim_matrix[i, i]) + 1
    true_sims = sim_matrix.diag().unsqueeze(1)  # shape (N, 1)
    # Compare each row with its own true similarity
    ranks = (sim_matrix > true_sims).sum(dim=1) + 1  # shape (N,)
    mrr = (1.0 / ranks.float()).mean().item()
    
    return mrr, mean_cos_sim

def compare_latent_spaces(embeddings_list, indices_list, compute_mrr = False, verbose=True):
    """
    Compares latent spaces by computing the Mean Reciprocal Rank (MRR) and cosine similarity 
    between pairs of embeddings.
    Args:
        embeddings_list (list): A list of embeddings where each element is an array of embeddings.
        indeces_list (list): A list of indices where each element is a an array of indeces corresponding to the embeddings.
    Returns:
        tuple: A tuple containing:
            - mrr_matrix (np.ndarray): A matrix of MRR values for each pair of embeddings.
            - mean_mrr (float): The mean MRR value across all pairs.
            - cos_sim_matrix (np.ndarray): A matrix of cosine similarity values for each pair of embeddings.
            - mean_cos_sim (float): The mean cosine similarity value across all pairs.
    """
    n = len(embeddings_list)
    mrr_matrix = np.zeros((n,n))
    cos_sim_matrix = np.zeros((n,n))

    iterator = tqdm(range(n), desc='Comparing latent spaces') if verbose else range(n)
    for i in iterator:
        for j in range(i,n):
            mrr, cos_sim = compute_latent_similarity(
                embeddings_list[i],
                indices_list[i],
                embeddings_list[j],
                indices_list[j],
                compute_mrr)
            mrr_matrix[i, j] = mrr
            cos_sim_matrix[i, j] = cos_sim
    
    # Exclude diagonal elements from the mean calculation
    mean_mrr = np.mean(mrr_matrix[np.triu_indices(n, k=1)])
    mean_cos_sim = np.mean(cos_sim_matrix[np.triu_indices(n, k=1)])

    print("\nSimilarity Results:")
    np.set_printoptions(precision=2, suppress=True)
    if compute_mrr:
        print(f"Mean Reciprocal Rank (MRR): {mean_mrr:.4f}")
        print("MRR Matrix:")
        print(mrr_matrix)

    print(f"\nMean Cosine Similarity: {mean_cos_sim:.4f}")
    print("Cosine Similarity Matrix:")
    print(cos_sim_matrix)

    return mrr_matrix, mean_mrr, cos_sim_matrix, mean_cos_sim