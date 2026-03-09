import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_dataloaders(data_dir, batch_size=32):

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transform
    )

    val_dataset = ImageFolder(
        root=f"{data_dir}/val",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, len(train_dataset.classes)



def get_subset_loader(dataset, fraction):

    n = len(dataset)

    indices = np.random.choice(
        n,
        int(n * fraction),
        replace=False
    )

    subset = Subset(dataset, indices)

    return subset