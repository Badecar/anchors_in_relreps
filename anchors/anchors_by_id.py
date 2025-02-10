from visualization import visualize_reconstruction_from_embedding, visualize_image_by_idx, visualize_reconstruction_by_id
import numpy as np

# Computing anchors and relative coordinates
def select_anchors_by_id(AE_list, embeddings_list, indices_list, desired_ids, dataset=None, show=False, device='cuda'):
    """
    Selects anchor embeddings based on the unique IDs from the dataset. Optionally shows
    the original images that correspond to the selected anchors.
    
    Args:
        embeddings (np.array): Array of shape [N, latent_dim] containing embeddings.
        all_ids (np.array): Array of shape [N] containing the unique dataset IDs.
        desired_ids (iterable): List or array of desired unique IDs to use as anchors.
        dataset (Dataset, optional): Dataset to retrieve the original images. 
            Must be indexable by the unique IDs. If show=True, this must be provided.
        show (bool, optional): If True, displays the images that were used as anchors.
    
    Returns:
        anchors (np.array): Array of selected anchor embeddings of shape [len(desired_ids), latent_dim].
    """
    anchor_set_list = []
    for AE, embeddings, all_ids in zip(AE_list, embeddings_list, indices_list):
        anchor_list = []
        for uid in desired_ids:
            # Find the position where the dataset id matches the desired id.
            idx = np.where(all_ids == uid)[0]
            if idx.size == 0:
                raise ValueError(f"ID {uid} not found in the obtained indices.")
            anchor_list.append(embeddings[idx[0]])
            
            # If show flag is set, display the corresponding image from the dataset and the reconstruction
            if show:
                visualize_image_by_idx(uid,dataset,use_flattened=True)
                visualize_reconstruction_from_embedding(embeddings[idx[0]],AE,device)
        
        anchor_set_list.append(np.stack(anchor_list))
    return anchor_set_list