from models import RelRepTrainer
import torch.nn as nn

def relrep_loss(anchor_num, anchors_list, num_epochs, AE_list, train_loader, test_loader, device, acc_list, train_loss_AE, test_loss_AE, head_type='decoder', distance_measure='cosine', lr=1e-3, verbose=True):
    """
    Trains a model on relative representations by attaching a suitable head (decoder or classifier)
    to a base autoencoder and then fitting it with provided training and test data.
    Parameters:
        anchor_num (int): Number of anchors in the relative representation.
        anchors_list (list): List of anchor values used by the trainer.
        num_epochs (int): Number of training epochs.
        AE_list (list): List containing autoencoder model(s); the first is used as the base model.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the training on.
        acc_list (list): List of accuracies from the autoencoder training.
        train_loss_AE (list): Training losses from the autoencoder phase.
        test_loss_AE (list): Test losses from the autoencoder phase.
        head_type (str, optional): Specifies whether to use a 'decoder' or 'classifier' head (default is 'decoder').
        distance_measure (str, optional): Distance metric used during training (default is 'cosine').
        lr (float, optional): Learning rate for training (default is 1e-3).
        verbose (bool, optional): If True, prints training progress (default is True).
    Returns:
        None
    """

    print("\nTraining on relative representations...")

    if head_type == 'decoder':
        # For MNIST, the reconstructed image is 28*28=784 dimensional.
        # Change this to fit the head of the model we are using
        head = nn.Sequential(
            nn.Linear(anchor_num, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )
    else:
        # Or this
        head = nn.Sequential(
            nn.Linear(anchor_num, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 10)
        )
        
    # Instantiate the trainer
    # Set head_type to 'decoder' (or 'classifier' if you're using a classifier head)
    trainer = RelRepTrainer(
        base_model=AE_list[0],
        head=head,
        anchors=anchors_list,
        distance_measure=distance_measure,
        head_type=head_type, # Change to 'classifier' for a classifier head
        device=device
    )

    # Train the head using the fit method
    fit_results = trainer.fit(
        train_loader,
        test_loader,
        num_epochs,
        lr=lr,
        verbose=verbose)

    train_loss_relrepfit, test_loss_relrepfit, test_accuracies = fit_results

    # Print the full list of losses per epoch
    print("\nFull Autoencoder Training Losses per Epoch:")
    print("Train Losses:", train_loss_AE)
    print("Test Losses:", test_loss_AE)
    if trainer.head_type == 'classifier':
        print("AE Accuracy:", acc_list[0])

    print("\nDecoder-Only Training Losses per Epoch (using relative representations):")
    print("Train Losses:", train_loss_relrepfit)
    print("Test Losses:", test_loss_relrepfit)
    if trainer.head_type == 'classifier':
        print("Test Accuracy:", test_accuracies[-1])