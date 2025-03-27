from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import torch
import random

def train_rel_decoder(epochs, hidden_dims, rel_model, model_list, relrep_list, idx_list, loader, device, nr_runs, show=True, verbose=True):
    """
    Train the relative decoder using zero-shot stitching on a validation set.
    This function performs the following steps:
    1. Validates the first autoencoder model from `model_list` using the provided `loader`.
    2. Processes the relative representations and images from the dataset, ensuring that images are in tensor form and sorted by a unique index.
    3. Constructs a TensorDataset from the first relative representation in `relrep_list` and the corresponding target images.
    4. Instantiates a relative decoder model using the provided `rel_model` constructor with parameters derived from the first autoencoder model.
    5. Trains the relative decoder on the constructed dataset for a fixed number of epochs.
    Parameters:
        rel_model (callable): A constructor for the relative decoder model, which should accept parameters such as
                              relative output dimension, encoder output shape, and number of channels.
        model_list (list): A list of pre-trained models; the first model is used for validation and to provide configuration details.
        relrep_list (list): A list containing relative representations; the first element is used for training the relative decoder.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset, used both for autoencoder validation and for constructing the
                                                training dataset for the relative decoder.
        device (torch.device): The device (CPU or GPU) on which the models and data will be placed for computation.
    Returns:
        tuple: A tuple containing:
            - rel_decoder (torch.nn.Module): The trained relative decoder model.
            - train_losses (list): A list of training losses recorded during the training process.
            - val_losses (list): A list of validation losses recorded during the training process.
    """
    

    print("Performing zero-shot stitching")
    # 1. Regular autoencoder validation on the first AE
    mse_reg, mse_std_reg = model_list[0].validate(loader, device=device)
    print("Regular AE Validation MSE: {:.5f} ± {:.5f}".format(mse_reg, mse_std_reg))

    # 2. Train the relative decoder using the relative coordinates and the validation set
    first_relrep = torch.tensor(relrep_list[0])

    # Build target_images using the dataset’s order
    to_tensor = transforms.ToTensor()
    target_images = []
    for i in range(len(loader.dataset)):
        img, _ = loader.dataset[i]
        # In case img is not already a tensor, convert it
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)
        img_flat = img.view(-1)
        target_images.append(img_flat)

    target_images = torch.stack(target_images, dim=0)

    # Sort the loader dataset by unique index to match the relrep order
    sorted_dataset = sorted(loader.dataset, key=lambda item: item[1][0])
    target_images = []
    for img, label in sorted_dataset:
        if not isinstance(img, torch.Tensor): # Convert to tensor if needed
            img = to_tensor(img)
        target_images.append(img.view(-1))
    target_images_tensor = torch.stack(target_images, dim=0)

    # Build the DataLoader for the relative decoder
    rel_decoder_dataset = TensorDataset(first_relrep, target_images_tensor)
    rel_decoder_loader = DataLoader(rel_decoder_dataset, batch_size=256, shuffle=True)

    # Instantiate the zero-shot relative decoder
    rel_decoder = rel_model(
        relative_output_dim=first_relrep.size(1),
        encoder_out_shape=model_list[0].encoder_out_shape,  # expected conv feature map shape
        n_channels=model_list[0].image_shape[0],             # e.g., 1 for MNIST
        hidden_dims = hidden_dims
    )
    rel_decoder.to(device)

    # Train the relative decoder.
    train_losses, val_losses = rel_decoder.fit(
        train_loader=rel_decoder_loader,
        test_loader=rel_decoder_loader,
        num_epochs=epochs,
        lr=1e-3,
        device=device,
        verbose=verbose
    )

    #NOTE: We do zero-shot here
    if show:
        sample_indices = random.sample(range(len(loader.dataset)), 16)
        sample_indices = list(range(20, 37))
        plot_reconstructions(
            rel_decoder=rel_decoder,
            relreps_list=relrep_list,
            unique_ids_list=idx_list,
            dataset=loader.dataset,
            sample_indices=sample_indices,
            device=device
        )

    return rel_decoder, train_losses, val_losses

def plot_reconstructions(rel_decoder, relreps_list, unique_ids_list, dataset, sample_indices, device):
    """
    Plots a grid of images with the ground truth on the top row followed by one row of decoded images for each AE run.
    Each column corresponds to one of the sample indices.
    
    Args:
        rel_decoder (torch.nn.Module): The trained relative decoder model.
        relreps_list (list): A list where each element is the relative representations 
                                for a given AE run (list or tensor).
        unique_ids_list (list): A list where each element is a list of unique ids corresponding to the dataset order 
                                for a given AE run. Ground truth is determined using the first AE run's unique ids.
        dataset (Dataset or DataLoader): The dataset where each item is (image, (uid, label)).
        sample_indices (list): A list of integer indices for samples to plot.
        device (torch.device): The device (CPU or GPU) on which computations are performed.
    """
    import matplotlib.pyplot as plt
    to_tensor = transforms.ToTensor()

    # --- Compute Ground Truth Images ---
    ground_truth_images = []
    first_unique_ids = unique_ids_list[0]  # use the first AE run's unique ids for ground truth
    for idx in sample_indices:
        expected_uid = first_unique_ids[idx]
        target_found = False
        for data in dataset:
            image, info = data
            uid, _ = info
            # Handle possible tensor vs scalar uid.
            if isinstance(uid, torch.Tensor):
                uid_val = uid.item() if uid.numel() == 1 else uid.tolist()
            else:
                uid_val = uid
            if uid_val == expected_uid:
                if not isinstance(image, torch.Tensor):
                    image = to_tensor(image)
                # Reshape image if necessary.
                if image.numel() == 28 * 28:
                    image = image.view(28, 28)
                elif image.dim() > 2 and image.shape[0] == 1:
                    image = image.squeeze(0)
                ground_truth_images.append(image.cpu())
                target_found = True
                break
        if not target_found:
            raise ValueError(f"Image with unique id {expected_uid} not found in the dataset.")

    # --- Decode Images for Each AE Run ---
    reconstructions_all = []
    for run_idx, relreps in enumerate(relreps_list):
        row_recons = []
        for idx in sample_indices:
            rep = relreps[idx]
            if not isinstance(rep, torch.Tensor):
                rep = torch.tensor(rep)
            rep = rep.unsqueeze(0).to(device)
            decoded = rel_decoder(rep)
            decoded_img = decoded.cpu().detach().view(28, 28)
            row_recons.append(decoded_img)
        reconstructions_all.append(row_recons)

    # --- Plotting ---
    n_cols = len(sample_indices)
    n_rows = 1 + len(relreps_list)  # one row for ground truth, one for each AE run.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Ensure axes is 2D.
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    
    # Top row: Ground Truth.
    for i in range(n_cols):
        axes[0][i].imshow(ground_truth_images[i], cmap="gray")
        axes[0][i].set_title("Ground Truth")
        axes[0][i].axis("off")
    
    # Subsequent rows: Decoded images from each AE run.
    for row in range(len(relreps_list)):
        for i in range(n_cols):
            axes[row + 1][i].imshow(reconstructions_all[row][i], cmap="gray")
            axes[row + 1][i].set_title(f"Decoded AE {row}")
            axes[row + 1][i].axis("off")
    
    plt.tight_layout()
    plt.show()