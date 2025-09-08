"""
Contains functionality for creating PyTorch DataLoader's for
image classification data.
"""
import os 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS=os.cpu_count()

def create_Dataloaders(
    train_dir:str,
    test_dir:str,
    transform:transforms.Compose,
    batch_size:int,
    num_workers:int=NUM_WORKERS
):
    """
    Creating training and testing dataloaders

    Returns:
    A tuple of (train_dataloader,test_dataloader,class_names).
    Where classnames is list of target classes
    Example_Usage:
    train_dataloaders,test_dataloaders,class_names=create_Dataloaders(train_dir,
    test_dir,
    transform,
    1)
    """
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=transform) # transforms to perform on data (images)
                                  
    test_data = datasets.ImageFolder(root=test_dir, 
                                 transform=transform)

    train_dataloader=DataLoader(dataset=train_data,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=True,
                               pin_memory=True)
    
    test_dataloader=DataLoader(dataset=test_data,
                               batch_size=batch_size,
                               num_workers=num_workers,
                               shuffle=False,
                               pin_memory=True)
    class_names=train_data.classes
    return train_dataloader,test_dataloader,class_names
     
