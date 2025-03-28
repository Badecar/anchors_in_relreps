import numpy as np
from scipy.spatial.distance import pdist


def compute_relative_coordinates_cossim(embeddings_list, anchors_list, flatten=False):
    """
    Transforms embeddings into a relative coordinate system based on provided anchors.
    This implementation normalizes both embeddings and anchors, and then computes
    cosine similarities between each embedding and each anchor.

    Args:
        embeddings (Tensor or np.array): Array of shape [N, latent_dim].
        anchors (Tensor or np.array): Array of shape [A, latent_dim], where A = number of anchors.
        flatten (bool): If True, each embedding is flattened before processing.

    Returns:
        relative_embeds (np.array): Array of shape [N, A] where each element is the cosine similarity
                                     between an embedding and an anchor.
    """
    relative_reps_outer = []
    for embeddings, anchors in zip(embeddings_list, anchors_list):
        # Normalize embeddings and anchors along the latent_dim axis
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        anchors_norm = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
        
        relative_reps_inner = []
        
        for embedding in embeddings_norm:
            if flatten:
                embedding = embedding.flatten()

            # Compute cosine similarity by dot product with each normalized anchor
            reletive_rep = np.array([np.dot(embedding, anchor) for anchor in anchors_norm])
            relative_reps_inner.append(reletive_rep)
        
        relative_reps_outer.append(np.array(relative_reps_inner))
    return relative_reps_outer


def compute_relative_coordinates_euclidean(embeddings_list, anchors_list, flatten=False):
    relative_reps_outer = []
    for embeddings, anchors in zip(embeddings_list, anchors_list):
        diff = np.array([np.linalg.norm(embeddings - anchor, axis=1) for anchor in anchors]).T
        relative_reps_outer.append(diff)
    return relative_reps_outer


def compute_relative_coordinates_mahalanobis(embeddings_list, anchors_list, inv_cov=None, epsilon=1e-6):
    """
    Computes the relative representation based on the Mahalanobis distance.
    
    For each pair of embeddings (shape [N, latent_dim]) and anchors (shape [A, latent_dim])
    in the provided lists, computes the pairwise Mahalanobis distances:
  
        d(x, a) = sqrt((x-a)^T * inv_cov * (x-a))
  
    If inv_cov is not provided, it is computed from the embeddings using the sample covariance.

    Args:
        embeddings_list (list of np.array): List of embeddings arrays, each with shape [N, d].
        anchors_list (list of np.array): List of anchor arrays, each with shape [A, d].
        inv_cov (np.array, optional): Precomputed inverse covariance matrix of shape [d, d]. 
            If None, it is computed per run.
        epsilon (float, optional): Small constant for numerical stability when inverting.

    Returns:
        List of np.array: Each array has shape [N, A] containing the negative Mahalanobis distances.
                          (Negative distances so that closer points have higher similarity.)
    """
    relative_reps_outer = []
    for embeddings, anchors in zip(embeddings_list, anchors_list):
        # If inv_cov not provided, compute it from the embeddings.
        if inv_cov is None:
            # np.cov expects variables in rows so set rowvar=False for data in columns.
            cov = np.cov(embeddings, rowvar=False)
            inv_cov_run = np.linalg.inv(cov + epsilon * np.eye(cov.shape[0]))
        else:
            inv_cov_run = inv_cov

        # Compute pairwise differences: shape [N, A, d]
        diff = embeddings[:, None, :] - anchors[None, :, :]
        # Compute squared Mahalanobis distances: for each [n, a]:
        # sum_{d,c} diff[n,a,d] * inv_cov_run[d,c] * diff[n,a,c]
        sq_dists = np.einsum("nad,dc,nac->na", diff, inv_cov_run, diff)
        # Add a small constant for numerical stability and take square root.
        dists = np.sqrt(sq_dists + 1e-8)
        # Return negative distances so that a smaller distance gives a higher similarity.
        rel_rep = -dists
        relative_reps_outer.append(rel_rep)
    return relative_reps_outer


def encode_relative_by_index(index, embeddings, anchors, flatten=False):
    """
    Computes the relative representation for a given data point index.
    
    Args:
        index (int): Index of the data point.
        embeddings (np.ndarray): Array of shape [N, latent_dim] containing latent embeddings.
        anchors (np.ndarray): Array of shape [A, latent_dim] containing anchor embeddings.
        flatten (bool): If True, will flatten the embeddings before processing.

    Returns:
        np.ndarray: Relative representation vector of shape [A,], where each element is the cosine 
                    similarity between the data point's latent embedding and an anchor.
    """
    # Retrieve the latent embedding for the specified index.
    embedding = embeddings[index] # Assuming that the embeddings are sorted by index already (which they should be)
    
    if flatten:
        embedding = embedding.flatten()
    
    # Normalize the embedding vector.
    embedding_norm = embedding / np.linalg.norm(embedding)
    
    # Normalize the anchors along the latent_dim axis.
    anchors_norm = anchors / np.linalg.norm(anchors, axis=1, keepdims=True)
    
    # Compute cosine similarity between the normalized embedding and each normalized anchor.
    rel_rep = np.dot(anchors_norm, embedding_norm)
    
    return rel_rep