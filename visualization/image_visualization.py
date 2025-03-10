import torch
import matplotlib.pyplot as plt

# Visualization functions

def visualize_reconstruction_by_id(unique_id, autoencoder, dataset, device='cuda'):
    """
    Visualizes the original image and its reconstruction from the autoencoder
    corresponding to a given unique dataset ID.
    
    Args:
        unique_id (int): The unique index of the MNIST image to visualize.
        autoencoder (nn.Module): The trained autoencoder.
        dataset (Dataset): The IndexedMNIST dataset instance.
        device (str): 'cpu' or 'cuda'.
    """
    autoencoder.eval()
    
    # Retrieve the image using the unique_id from the dataset.
    # Since the dataset is IndexedMNIST, its __getitem__ returns (image, (uid, label))
    image, (uid, label) = dataset[unique_id]
    if uid != unique_id:
        raise ValueError(f"Mismatch: expected unique id {unique_id} but got {uid}")
    
    # Prepare the image tensor by moving it to the correct device and adding a batch dimension.
    image_tensor = image.to(device).unsqueeze(0)
    
    with torch.no_grad():
        reconstruction = autoencoder(image_tensor)
    
    # Reshape the flattened image tensors into 28x28 for visualization.
    original = image_tensor.view(28, 28).cpu()
    reconstructed = reconstruction.view(28, 28).cpu()
    
    # Plot original and reconstructed images side by side.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original (ID: {unique_id}, Label: {label})')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title(f'Reconstruction (ID: {unique_id})')
    axes[1].axis('off')
    
    plt.show()


def visualize_image_by_idx(idx, dataset, use_flattened=True):
    """
    Visualizes a specific MNIST image given its unique index from the dataset.
    
    Args:
        idx (int): The unique index of the image.
        dataset (Dataset): The MNIST dataset instance.
        use_flattened (bool): True if the stored image is flattened.
                             If True, the image will be reshaped to (28,28) for display.
    """
    # Get the image and label from the dataset
    image, label = dataset[idx]
    
    # If the image is flattened, reshape it for visualization
    if use_flattened:
        image = image.view(28, 28)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(image.cpu(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()


def visualize_reconstruction_from_embedding(embedding, autoencoder, device='cuda'):
    """
    Visualizes the image obtained by decoding a given anchor embedding.
    
    Args:
        anchor_embedding (np.array or Tensor): The latent vector for the anchor of shape [latent_dim].
        autoencoder (nn.Module): The trained autoencoder.
        device (str): 'cpu' or 'cuda'.
    """
    autoencoder.eval()
    
    # Convert anchor_embedding to a Tensor if it's not already, and ensure it has a batch-dimension.
    if not torch.is_tensor(embedding):
        embedding = torch.tensor(embedding, dtype=torch.float)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    
    embedding = embedding.to(device)
    
    with torch.no_grad():
        # Use the decode function to get the reconstructed image.
        decoded = autoencoder.decode(embedding)
    
    # Reshape the flattened image tensor (assuming MNIST images => 28x28)
    image = decoded.view(28, 28).cpu().numpy()
    
    plt.figure(figsize=(3,3))
    plt.imshow(image, cmap='gray')
    plt.title("Decoded Anchor")
    plt.axis("off")
    plt.show()

def visualize_reconstruction_from_embedding_with_decoder(embedding, decoder, device='cuda'):
    """
    Visualizes the image obtained by decoding a given latent embedding using the provided decoder/head.
    
    Args:
        embedding (np.array or Tensor): The latent vector of shape [latent_dim].
        decoder (nn.Module): The trained decoder or relative head.
        device (str): 'cpu' or 'cuda'.
    """

    decoder.eval()

    # Convert embedding to a Tensor if not already and ensure it has a batch dimension.
    if not torch.is_tensor(embedding):
        embedding = torch.tensor(embedding, dtype=torch.float)
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    
    embedding = embedding.to(device)

    with torch.no_grad():
        # Use the decoder/head to get the output.
        decoded = decoder(embedding)
    
    # Reshape the flattened output to (28,28) assuming MNIST images.
    image = decoded.view(28, 28).cpu().numpy()
    
    plt.figure(figsize=(3,3))
    plt.imshow(image, cmap='gray')
    plt.title("Decoded Reconstruction")
    plt.axis("off")
    plt.show()