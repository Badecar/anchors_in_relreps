import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, '..'))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
from utils import *
import torch
import random
import numpy as np
from models import *
import torch.nn as nn
from data import *
from visualization import *
from anchors import *
from relreps import *
from tqdm import tqdm
from sklearn.decomposition import PCA
def compute_mrr_func(ft_matrix, w2v_matrix):
    """
    Computes the Mean Reciprocal Rank (MRR) between fastText and Word2Vec embeddings.
    It assumes that for each word, the fastText embedding (query) and the corresponding 
    Word2Vec embedding (target) are aligned at the same index.

    Args:
        ft_matrix (np.ndarray): fastText embedding matrix of shape (N, D).
        w2v_matrix (np.ndarray): Word2Vec embedding matrix of shape (N, D).

    Returns:
        mrr (float): The Mean Reciprocal Rank.
    """
    N = ft_matrix.shape[0]
    
    # Normalize both matrices so that cosine similarity is just dot product.
    ft_norms = np.linalg.norm(ft_matrix, axis=1, keepdims=True)
    w2v_norms = np.linalg.norm(w2v_matrix, axis=1, keepdims=True)
    
    ft_normalized = ft_matrix / (ft_norms + 1e-10)
    w2v_normalized = w2v_matrix / (w2v_norms + 1e-10)
    
    reciprocal_ranks = []
    
    for i in range(N):
        # Use fastText vector at index i as the query.
        query = ft_normalized[i]  # shape: (D,)
        # Compute cosine similarity with all Word2Vec embeddings.
        similarities = np.dot(w2v_normalized, query)  # shape: (N,)
        # Sort indices in descending order of similarity.
        sorted_indices = np.argsort(-similarities)
        # Find the rank of the correct Word2Vec vector (at index i).
        rank = np.where(sorted_indices == i)[0][0] + 1  # Convert to 1-indexed rank.
        reciprocal_ranks.append(1.0 / rank)
    
    mrr = np.mean(reciprocal_ranks)
    return mrr

def create_visualization_subset_indices(emb_list, idx_list, topn=200, num_pivots=4):
    """
    Given a NumPy matrix 'wv' of word embeddings and a corresponding 'vocab' list,
    randomly selects 'num_pivots' pivot indices and, for each pivot, finds the top 'topn'
    neighbors (by cosine similarity). Returns:
      - pivot_indices: indices of the pivot words.
      - subset_indices: sorted union of pivot indices and their top 'topn' neighbors.
      - labels: a list of cluster labels (0,1,...,num_pivots-1) corresponding to each index in subset_indices,
                indicating which pivot the word is most similar to.
    
    Args:
        wv (np.ndarray): A NumPy array of shape (N, D), where N is the number of words.
        vocab (list): A list of words corresponding to the rows of wv.
        topn (int): Number of neighbors to retrieve per pivot.
        num_pivots (int): Number of random pivot words to select.
        seed (int): Seed for reproducibility.
        
    Returns:
        pivot_indices (list of int): List of pivot indices.
        subset_indices (list of int): Sorted list of indices corresponding to the union
                                      of pivot indices and their top 'topn' neighbors.
        labels (list of int): List of cluster labels for each index in subset_indices, 
                              determined by which pivot (0 to num_pivots-1) is most similar.
    """
    wv = emb_list[0]
    N, _ = wv.shape

    # 1. Select random pivot indices.
    pivot_indices = np.load("2D_AE_relrep/pivot_indices.npy")
    pivot_points = [wv[i] for i in pivot_indices]
    subset_indices = []
    labels = []
    # 2. For each pivot, retrieve its top 'topn' neighbors
    for i, pivot_point in enumerate(pivot_points):
        distances = np.linalg.norm(wv - pivot_point, axis=1)
        # Sort indices by ascending distance (closest first).
        sorted_indices = np.argsort(distances)
        
        neighbors = []
        for idx in sorted_indices:
            neighbors.append(idx)
            labels.append(i)
            if len(neighbors) >= topn:
                break
        
        subset_indices.extend(neighbors)
    
    
    pivot_indices, subset_indices, labels = np.array(pivot_indices), np.array(subset_indices), np.array(labels)
    print(pivot_indices.shape, subset_indices.shape, labels.shape)
    return pivot_indices, subset_indices, labels


# For reproducibility and consistency across runs, we set a seed
seed = 42
set_random_seeds(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}: {torch.cuda.get_device_name(0)}")

# Load data
train_loader, test_loader, val_loader = load_mnist_data()
loader = val_loader
use_small_dataset = False
### MAKE SURE THAT WE ARE USING THE SAME DATA FOR ALL FUNCTIONS ###

# TODO: CURRENTLY USING OLD VERSION OF THE AE FOR TESTING. NEED TO UPDATE TO NEW VERSION

### PARAMETERS ###
model = Autoencoder #AEClassifier or Autoencoder
head_type = 'reconstructor'    #reconstructor or classifier
distance_measure = 'cosine'   #cosine or euclidean
load_saved = False       # Load saved embeddings from previous runs (from models/saved_embeddings)
save_run = False        # Save embeddings from current run
anchor_num = 300
nr_runs = 3            # If load_saved: Must be <= number of saved runs for the dim
number_of_datapoints = 20_000

# Train parameters
num_epochs = 30
lr = 1e-3

# Hyperparameters for anchor selection
coverage_w = 0.92
diversity_w = 0.08
exponent = 1

# Post-processing
plot_results = True
zero_shot = False
compute_mrr = True      # Only set true if you have >32GB of RAM, and very low dim
compute_similarity = True
compute_relrep_loss = False # Can only compute if not loading from save
### ###
ft_embeddings = np.load(r"datasets\word_embeddings\ft_matrix.npy")
w2v_embeddings = np.load(r"datasets\word_embeddings\w2v_matrix.npy")
data_point_idx = np.random.choice(np.arange(len(ft_embeddings)), size=number_of_datapoints)
emb_list = [ft_embeddings[data_point_idx], w2v_embeddings[data_point_idx]]
idx_list = [range(len(emb)) for emb in emb_list]
# Find anchors and compute relative coordinates
random_anchor_ids = np.array(random.sample(list(idx_list[0]), anchor_num))
rand_anchors_list = [emb[random_anchor_ids] for emb in emb_list]

# TODO: Instead of softmax, then pass the size of the weights of P into the loss. Average of the sum over each column (A)
# Optimize anchors and compute P_anchors_list
# anchor_selector, P_anchors_list = get_optimized_anchors(
#     emb = emb_list,
#     anchor_num=anchor_num,
#     epochs=50,
#     lr=1e-1,
#     coverage_weight=coverage_w,
#     diversity_weight=diversity_w,
#     exponent=exponent,
#     verbose=False,
#     device=device
# )




# USING RANDOM AND COSINE FOR ZERO-SHOT STICHING TESTS
anch_list = rand_anchors_list
relrep_list = compute_relative_coordinates_cossim(emb_list, anch_list)


# create visualization subset
visualization_subset_indices_pivot, visualization_subset_indices_pivot_data, label = create_visualization_subset_indices(emb_list, idx_list)

labels_list = [label for i in range(len(emb_list))]
emb_list_vis = [emb[visualization_subset_indices_pivot_data] for emb in emb_list]
relrep_list_vis = [relrep[visualization_subset_indices_pivot_data] for relrep in relrep_list]


### Plotting ###
if plot_results:
    # Plot encodings side by side
    # is_relrep = True
    # if is_relrep:
    #     print("Plotting relrep")
    #     plot_data_list(relrep_list, labels_list, do_pca=False, is_relrep=is_relrep, anchors_list=anch_rel)
    # else:
    #     print("Plotting absolute embeddings")
    #     plot_data_list(emb_list, labels_list, do_pca=True, is_relrep=is_relrep, anchors_list=rand_anchors_list)
    print("Plotting absolute embeddings")
    plot_data_list(emb_list_vis, labels_list, do_pca=False, is_relrep=False, anchors_list=None)
    print("Plotting relrep")
    plot_data_list(relrep_list_vis, labels_list, do_pca=False, is_relrep=False, anchors_list=None)

#mrr_matrix, mean_mrr, cos_sim_matrix, mean_cos_sim = compare_latent_spaces(emb_list, [np.arange(len([0])), np.arange(len([1]))], compute_mrr=compute_mrr, verbose=False)

mrr = compute_mrr_func(emb_list[0], emb_list[1])
print("abs: ", mrr)
mrr = compute_mrr_func(relrep_list[0], relrep_list[1])
print("relrep: ", mrr)
