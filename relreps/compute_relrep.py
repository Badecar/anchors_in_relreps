import numpy as np

def compute_relative_coordinates(embeddings_list, anchors_list, flatten=False):
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