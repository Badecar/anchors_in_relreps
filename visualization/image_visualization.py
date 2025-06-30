import torch
import matplotlib.pyplot as plt

# Visualization functions

def visualize_reconstruction_by_id(unique_id, autoencoder, dataset, device='cuda'):
    """
    Visualizes the original image and its reconstruction from the autoencoder
    corresponding to a given unique dataset ID by searching the DataLoader for a match.
    
    Args:
        unique_id (int): The unique id of the MNIST image.
        autoencoder (nn.Module): The trained autoencoder.
        dataset (DataLoader or Dataset): The dataset where each item is (image, (uid, label)).
        device (str): 'cpu' or 'cuda'.
    """
    import matplotlib.pyplot as plt
    autoencoder.eval()
    
    # Iterate over the dataset to find the matching unique_id.
    found = False
    for data in dataset:
        image, info = data
        uid, label = info
        # Handle batched uid values.
        if isinstance(uid, torch.Tensor) and uid.numel() > 1:
            uid_list = uid.tolist()
            label_list = label.tolist() if isinstance(label, torch.Tensor) and label.numel() > 1 else label
            for i, u in enumerate(uid_list):
                if u == unique_id:
                    image_entry = image[i]
                    label_entry = label_list[i]
                    found = True
                    break
            if found:
                break
        else:
            uid_val = uid.item() if isinstance(uid, torch.Tensor) else uid
            if uid_val == unique_id:
                image_entry = image
                label_entry = label
                found = True
                break

    if not found:
        raise ValueError(f"Image with unique id {unique_id} not found in the dataset.")
    
    # Move the image to the correct device and add a batch dimension.
    image_tensor = image_entry.to(device).unsqueeze(0)
    
    with torch.no_grad():
        reconstruction = autoencoder(image_tensor)
    
    # Reshape the flattened tensors into 28x28 images.
    original = image_tensor.view(28, 28).cpu()
    reconstructed = reconstruction.view(28, 28).cpu()
    
    # Plot original and reconstructed images side by side.
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original (ID: {unique_id}, Label: {label_entry})')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title(f'Reconstruction (ID: {unique_id})')
    axes[1].axis('off')
    
    plt.show()


def visualize_image_by_idx(idx, dataset, use_flattened=True):
    """
    Visualizes a specific MNIST image given its unique id from the dataset.
    
    Instead of assuming the image is at position idx,
    this function iterates through the dataset to find the entry where the unique id matches idx.
    
    Args:
        idx (int): The unique id of the image.
        dataset (Dataset or DataLoader): The MNIST dataset instance where each item is expected to be (image, (uid, label)).
        use_flattened (bool): True if the stored image is flattened.
                             If True, the image will be reshaped to (28,28) for display.
    """
    for data in dataset:
        image, info = data
        uid, label = info
        # If uid is a batch tensor, iterate over its elements.
        if isinstance(uid, torch.Tensor) and uid.numel() > 1:
            uid_list = uid.tolist()
            # Also convert label to list if needed.
            label_list = label.tolist() if isinstance(label, torch.Tensor) and label.numel() > 1 else label
            for i, u in enumerate(uid_list):
                if u == idx:
                    im = image[i]
                    if use_flattened:
                        im = im.view(28, 28)
                    plt.figure(figsize=(4, 4))
                    plt.imshow(im.cpu(), cmap='gray')
                    plt.title(f"Label: {label_list[i]}")
                    plt.axis("off")
                    plt.show()
                    return
        else:
            # Ensure uid is a scalar
            if isinstance(uid, torch.Tensor):
                uid = uid.item()
            if uid == idx:
                if use_flattened:
                    image = image.view(28, 28)
                plt.figure(figsize=(4, 4))
                plt.imshow(image.cpu(), cmap='gray')
                plt.title(f"Label: {label}")
                plt.axis("off")
                plt.show()
                return
    raise ValueError(f"Image with unique id {idx} not found in the dataset.")

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