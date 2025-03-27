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


def compute_relative_coordinates_cossim_non_normal(embeddings_list, anchors_list, flatten=False):
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
        embeddings_norm = embeddings
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