import numpy as np
import torch
import random

# For reproducibility and consistency across runs, we will set a seed
def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_smaller_dataset(embeddings_list, indices_list, labels_list, samples_per_class=10):
    """
    Create a balanced subset for each trial by selecting a fixed number of embeddings per class.
    Ensures a consistent ordering by first sorting by label and then by the original indices.
    
    Args:
        embeddings_list (list of np.ndarray): Each element is an array of shape (N, D) embeddings.
        indices_list (list of np.ndarray): Each element is an array of shape (N,) with original dataset indices.
        labels_list (list of np.ndarray): Each element is an array of shape (N,) with class labels.
        samples_per_class (int): Number of samples to select per class (if available).
    
    Returns:
        tuple: A tuple of three lists:
            - balanced_embeddings_list (list of np.ndarray): Balanced embeddings for each trial.
            - balanced_indices_list (list of np.ndarray): Corresponding indices.
            - balanced_labels_list (list of np.ndarray): Corresponding labels.
    """
    balanced_embeddings_list = []
    balanced_indices_list = []
    balanced_labels_list = []

    # Process each trial separately.
    for emb, idx, labs in zip(embeddings_list, indices_list, labels_list):
        # First sort according to label then index so that the order is the same across trials.
        mask = np.argsort(idx)
        emb_sorted = emb[mask]
        idx_sorted = idx[mask]
        labs_sorted = labs[mask]

        unique_labels = np.unique(labs_sorted)
        balanced_emb = []
        balanced_idx = []
        balanced_lab = []

        for label in unique_labels:
            # Find all positions for the current label.
            label_mask = (labs_sorted == label)
            positions = np.where(label_mask)[0]
            # Select samples_per_class entries (or all if there are fewer)
            selected = positions[:samples_per_class]
            balanced_emb.append(emb_sorted[selected])
            balanced_idx.append(idx_sorted[selected])
            balanced_lab.append(labs_sorted[selected])
            
        # Concatenate the selections from each class.
        balanced_emb = np.concatenate(balanced_emb, axis=0)
        balanced_idx = np.concatenate(balanced_idx, axis=0)
        balanced_lab = np.concatenate(balanced_lab, axis=0)

        # Final sorting by original indices for consistency.
        final_order = np.argsort(balanced_idx)
        balanced_embeddings_list.append(balanced_emb[final_order])
        balanced_indices_list.append(balanced_idx[final_order])
        balanced_labels_list.append(balanced_lab[final_order])
        
    return balanced_embeddings_list, balanced_indices_list, balanced_labels_list