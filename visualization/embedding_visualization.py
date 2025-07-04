import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Plotting
def fit_and_align_pca(data, ref_pca=None):
    """
    Fits PCA on 'data', then aligns its components with the reference PCA 
    so that signs are consistent. Returns (pca, data_pca), where pca
    is the fitted PCA object and data_pca is the 2D projection.

    If ref_pca is None, this PCA becomes the new reference.
    If ref_pca is not None, the new PCA is aligned (sign-flipped if needed)
    to match the reference's orientation.
    """
    pca = PCA(n_components=2, random_state=42)
    data_pca = pca.fit_transform(data)
    
    if ref_pca is None:
        # First time: no reference to align with, just return
        return pca, data_pca
    
    # Align the sign of the new components with the reference
    for i in range(2):
        dot_product = np.dot(pca.components_[i], ref_pca.components_[i])
        if dot_product < 0:
            pca.components_[i] = -pca.components_[i]
            data_pca[:, i]     = -data_pca[:, i]
    
    return data_pca


# def plot_data_list(data_list, labels_list, do_pca=True, ref_pca=None,
#                    is_relrep=True, anchors_list=None):
#     """
#     Plots multiple datasets side-by-side in subplots (1 row, len(data_list) columns).
    
#     data_list : list of np.ndarray
#         Each element is a dataset (shape [n_samples, n_features]).
#     label_list : list of np.ndarray
#         Each element is the label array for the corresponding dataset.
#     do_pca : bool
#         Whether to run PCA on the data. If True, we use fit_and_align_pca.
#     ref_pca : PCA or None
#         If None, the first dataset's PCA becomes the reference.
#         If not None, subsequent datasets align to this PCA orientation.
#     get_ref_pca : bool
#         If True, return the final PCA used for alignment (could be the first one).
#     is_relrep : bool
#         If True, changes the plot title to "Related Representations", else "AE Encodings".
#     """
    
#     n_plots = len(data_list)
#     fig, axs = plt.subplots(1, n_plots, figsize=(7*n_plots, 4), squeeze=False)
#     axs = axs.ravel()  # Flatten in case there's only 1 subplot
    
#     for i, (data, labels) in enumerate(zip(data_list, labels_list)):
#         if do_pca:
#             if ref_pca is None:
#                 pca, data_2d = fit_and_align_pca(data, ref_pca=ref_pca)
#             else:
#                 if is_relrep:
#                     data_2d = fit_and_align_pca(data, ref_pca=ref_pca)
#                 else:
#                     _, data_2d = fit_and_align_pca(data, ref_pca=None)

#             # If we didn't already have a reference, the first fitted pca becomes ref
#             if i == 0 and ref_pca is None:
#                 ref_pca = pca
#         else:
#             data_2d = data
        
#         scatter = axs[i].scatter(data_2d[:, 0], data_2d[:, 1],
#                                  c=labels, cmap='tab10', s=10, alpha=0.7)
#         if anchors_list is not None:
#             axs[i].scatter(anchors_list[i][:, 0], anchors_list[i][:, 1], s=25, marker="*", c='#000000')
#         # Optionally add a colorbar to each subplot
#         cb = fig.colorbar(scatter, ax=axs[i], ticks=range(10))
#         cb.set_label('Label')
        
#         axs[i].set_xlabel('PC 1')
#         axs[i].set_ylabel('PC 2')
#         if is_relrep:
#             axs[i].set_title(f'2D PCA of Relative Representations {i+1}')
#         else:
#             axs[i].set_title(f'2D PCA of AE Encodings {i+1}')
    
#     plt.tight_layout()
#     plt.show()


# Potentially fixed so anchors are plotted in the same PCA space as the data
def plot_data_list(data_list, labels_list, do_pca=True, ref_pca=None,
                   is_relrep=True, anchors_list=None, title=None, output_file='embeddings_plot.png'):
    """
    Plots multiple datasets side-by-side in subplots (1 row, len(data_list) columns).
    
    data_list : list of np.ndarray
        Each element is a dataset (shape [n_samples, n_features]).
    label_list : list of np.ndarray
        Each element is the label array for the corresponding dataset.
    do_pca : bool
        Whether to run PCA on the data. If True, we use fit_and_align_pca.
    ref_pca : PCA or None
        If None, the first dataset's PCA becomes the reference.
        If not None, subsequent datasets align to this PCA orientation.
    is_relrep : bool
        If True, changes the plot title to "Related Representations", else "AE Encodings".
    title: str or None
        Custom title for plots. If None, a default title is set.
    """
    
    n_plots = len(data_list)
    fig, axs = plt.subplots(1, n_plots, figsize=(7*n_plots, 4), squeeze=False)
    axs = axs.ravel()  # Flatten in case there's only 1 subplot
    
    for i, (data, labels) in enumerate(zip(data_list, labels_list)):
        if do_pca:
            # For the first dataset, or when ref_pca is None, fit PCA and use it as reference.
            if ref_pca is None:
                pca, data_2d = fit_and_align_pca(data, ref_pca=ref_pca)
                pca_current = pca
            else:
                if is_relrep:
                    # Align current data with the existing reference PCA.
                    data_2d = fit_and_align_pca(data, ref_pca=ref_pca)
                    pca_current = ref_pca
                else:
                    # Compute PCA independently for current data.
                    _, data_2d = fit_and_align_pca(data, ref_pca=None)
                    # No common PCA space; anchors (if any) are assumed pre-transformed.
                    pca_current = None

            if i == 0 and ref_pca is None:
                ref_pca = pca_current
        else:
            data_2d = data
            pca_current = None
        
        scatter = axs[i].scatter(data_2d[:, 0], data_2d[:, 1],
                                 c=labels, cmap='tab10', s=10, alpha=0.7)
        
        # Transform anchors using the same PCA if available
        if anchors_list is not None:
            anchors = anchors_list[i]
            if do_pca and (pca_current is not None):
                anchors_2d = pca_current.transform(anchors)
            else:
                anchors_2d = anchors
            axs[i].scatter(anchors_2d[:, 0], anchors_2d[:, 1], 
                           s=25, marker="*", c='#000000')
        
        # Optionally add a colorbar to each subplot
        cb = fig.colorbar(scatter, ax=axs[i], ticks=range(10))
        cb.set_label('Label')
        
        axs[i].set_xlabel('PC 1')
        axs[i].set_ylabel('PC 2')
        if title is None:
            axs[i].set_title(f'user to lazy to set title {i+1}')
        else:
            axs[i].set_title(f'2D PCA of AE Encodings {i+1}')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Latent plot saved to {output_file}")


def plot_3D_relreps(embeddings, labels):
    """
    Plots the AE latent embeddings color-coded by label.
    If embeddings are 2D, it makes a 2D plot.
    If embeddings are 3D, it creates an interactive 3D plot.

    Args:
        embeddings (np.array): Array of shape [N, 2] or [N, 3] containing latent embeddings.
        labels (np.array): Array of shape [N] containing the corresponding labels.
        title (str): Title for the plot.
    """
    if slice:
        condition = np.logical_not((embeddings[:, 0] > 0) & (embeddings[:, 1] > 0))
        embeddings = embeddings[condition]
        labels = labels[condition]
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2],
                             c=labels, cmap='tab10', s=10, alpha=0.7)
        # Create a colorbar. Note: use fig.colorbar and pass the mappable (scatter)
        cbar = fig.colorbar(scatter, ax=ax, ticks=range(10), shrink=0.5, aspect=10)
        cbar.set_label('Labels')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title("3D Relrep Plot")
        plt.show()