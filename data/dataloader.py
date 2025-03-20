import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Loading the MNIST dataset. Changing the __getitem_ function to give a tuple label with the number and a unique idx
# Custom MNIST dataset that returns image and a tuple (index, label)
class IndexedMNIST(datasets.MNIST):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # Return image and a tuple (unique index, label)
        return image, (index, label)

def load_mnist_data(batch_size=256, download=True, validation_split=0.1):
    """
    Loads and returns MNIST train, validation, and test DataLoaders with flattened images.
    Each sample's label is a tuple containing a unique index and the actual label.
    
    Args:
        batch_size (int): The batch size for the DataLoader.
        download (bool): Whether to download the dataset if not found.
        validation_split (float): The proportion of the training data to use for validation.
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects for the MNIST dataset.
    """
    # Define a transform to normalize and flatten the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # flatten image to a 1D vector of size 784
    ])
    
    # Load the training and test datasets
    train_dataset = IndexedMNIST(root='../datasets', train=True, transform=transform, download=download)
    test_dataset = IndexedMNIST(root='../datasets', train=False, transform=transform, download=download)
    
    # Split the training dataset into training and validation subsets
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create DataLoaders for the datasets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader

class IndexedFashionMNIST(datasets.FashionMNIST):
        def __getitem__(self, index):
            image, label = super().__getitem__(index)
            # Return image and a tuple (unique index, label)
            return image, (index, label)

def load_fashion_mnist_data(batch_size=256, download=True, validation_split=0.1):
    """
    Loads and returns FashionMNIST train, validation, and test DataLoaders with flattened images.
    Each sample's label is a tuple containing a unique index and the actual label.
    
    Args:
        batch_size (int): The batch size for the DataLoader.
        download (bool): Whether to download the dataset if not found.
        validation_split (float): The proportion of the training data to use for validation.
    
    Returns:
        train_loader, test_loader, val_loader: DataLoader objects for the FashionMNIST dataset.
    """
    # Define a transform to normalize and flatten the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.view(-1))  # flatten image to a 1D vector of size 784
    ])
    
    # Load the training and test datasets
    train_dataset = IndexedFashionMNIST(root='../datasets', train=True, transform=transform, download=download)
    test_dataset = IndexedFashionMNIST(root='../datasets', train=False, transform=transform, download=download)
    
    # Split the training dataset into training and validation subsets
    train_size = int((1 - validation_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Create DataLoaders for the datasets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
