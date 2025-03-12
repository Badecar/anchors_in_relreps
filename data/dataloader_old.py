import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Loading the MNIST dataset. Changing the __getitem_ function to give a touple label with the number and a unique idx
# Custom MNIST dataset that returns image and a tuple (index, label)
class IndexedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # Return image and a tuple (unique index, label)
        return image, (index, label)
        
def load_mnist_data(batch_size=256, download=True):
    """
    Loads and returns MNIST train and test DataLoaders with flattened images.
    Each sample's label is a tuple containing a unique index and the actual label.
    
    Args:
        batch_size (int): The batch size for the DataLoader.
        download (bool): Whether to download the dataset if not found.
    
    Returns:
        train_loader, test_loader: DataLoader objects for the MNIST dataset.
    """
    # Define a transform to normalize and flatten the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # flatten image to a 1D vector of size 784
    ])
    
    # Load the training and test datasets
    train_dataset = IndexedMNIST(root='../datasets', train=True, transform=transform, download=download)
    test_dataset = IndexedMNIST(root='../datasets', train=False, transform=transform, download=download)
    
    # Create DataLoaders for the datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader