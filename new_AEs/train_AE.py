import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Import sort_results from your data module and VanillaAE from ae.py
from data import sort_results
from .ae import VanillaAE
from .enumerations import Output  # used in loss_function for accessing keys

def train_VanillaAE(
    model_class,
    metadata,
    train_loader,
    test_loader,
    num_epochs=10,
    lr=1e-3,
    device='cuda',
    verbose=True,
    trials=1,
    latent_dim=10,
    hidden_layer=128
):
    """
    Trains VanillaAE over several trials and returns:
      AE_list, embeddings_list, indices_list, labels_list,
      train_loss (last trial), test_loss (last trial), acc_list.
    (acc_list entries are None since VanillaAE does not implement accuracy.)
    """
    AE_list = []
    embeddings_list = []
    indices_list = []
    labels_list = []
    acc_list = []  # VanillaAE does not implement accuracy, so we use None values
    
    last_trial_train_loss = None
    last_trial_test_loss = None

    for trial in range(trials):
        if verbose:
            print(f"Trial {trial+1} of {trials}")
        # Instantiate a new VanillaAE for the trial.
        # input_size is provided as (n_channels, height, width)
        model = model_class(
            metadata,
            input_size=(metadata.n_channels, metadata.height, metadata.width),
            latent_dim=latent_dim,
            hidden_dims=[hidden_layer]  # adjust as needed
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        model.train()
        for epoch in range(1, num_epochs + 1):
            epoch_train_loss = 0.0
            num_batches = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
                # Expecting batch to be either (images, labels) or (images, (indices, labels)).
                # We prepare a dictionary for the loss function (which expects key "image")
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                elif isinstance(batch, dict):
                    x = batch["image"]
                else:
                    x = batch
                    
                x = x.to(device)
                optimizer.zero_grad()
                model_out = model(x)
                # Create a data dict for loss_function
                loss_data = {"image": x}
                loss_dict = model.loss_function(model_out, loss_data)
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                num_batches += 1

            epoch_train_loss_avg = epoch_train_loss / num_batches if num_batches > 0 else 0.0

            # Evaluation loop for test loss
            model.eval()
            with torch.no_grad():
                epoch_test_loss = 0.0
                test_batches = 0
                for batch in test_loader:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                    elif isinstance(batch, dict):
                        x = batch["image"]
                    else:
                        x = batch
                    x = x.to(device)
                    model_out = model(x)
                    loss_data = {"image": x}
                    loss_dict = model.loss_function(model_out, loss_data)
                    epoch_test_loss += loss_dict["loss"].item()
                    test_batches += 1
                epoch_test_loss_avg = epoch_test_loss / test_batches if test_batches > 0 else 0.0

            if verbose:
                print(f"Epoch [{epoch}/{num_epochs}]: Train Loss: {epoch_train_loss_avg:.5f} | Test Loss: {epoch_test_loss_avg:.5f}")
            model.train()  # set back to training mode for next epoch

        last_trial_train_loss = epoch_train_loss_avg
        last_trial_test_loss = epoch_test_loss_avg

        # After training, extract latent embeddings using the train_loader
        embeddings, indices, labels = model.get_latent_embeddings(train_loader, device=device)
        # Sort the results based on indices for consistency
        emb_sorted, idx_sorted, labels_sorted = sort_results(
            embeddings.cpu().numpy(), indices.cpu(), labels.cpu().numpy()
        )
        embeddings_list.append(emb_sorted)
        indices_list.append(idx_sorted)
        labels_list.append(labels_sorted)
        AE_list.append(model)
        acc_list.append(None)

    # Return the trained models and metrics from the last trial
    return AE_list, embeddings_list, indices_list, labels_list, last_trial_train_loss, last_trial_test_loss, acc_list