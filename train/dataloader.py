import os
import sys

import torch
from torchvision import datasets
from torchvision import transforms


def get_ft_dataloader(train_folder_root, val_folder_root, sample_size=256,
                      crop_size=224, batch_size=32, num_workers=8):
    
    # compose transform
    train_trans = []
    train_trans.append(transforms.Resize(size=sample_size))
    train_trans.append(transforms.CenterCrop(size=sample_size))
    train_trans.append(transforms.RandomCrop(size=crop_size))
    train_trans.append(transforms.RandomHorizontalFlip())
    train_trans.append(transforms.ToTensor())
    train_trans = transforms.Compose(train_trans)

    val_trans = []
    val_trans.append(transforms.Resize(size=sample_size))
    val_trans.append(transforms.CenterCrop(size=crop_size))
    val_trans.append(transforms.ToTensor())
    val_trans = transforms.Compose(val_trans)

    # torchvision style datasets
    train_dataset = datasets.ImageFolder(root=train_folder_root, transform=train_trans)
    val_dataset = datasets.ImageFolder(root=val_folder_root, transform=val_trans)

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                                   shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, 
                                                   shuffle=True, num_workers=num_workers)
    return train_dataloader, val_dataloader


def get_atte_dataloader(train_folder_root, val_folder_root, sample_size=900,
                        crop_size=720, batch_size=32, num_workers=8):
    
    # compose transform
    train_trans = []
    train_trans.append(transforms.Resize(size=sample_size))
    train_trans.append(transforms.CenterCrop(size=sample_size))
    train_trans.append(transforms.RandomCrop(size=crop_size))
    train_trans.append(transforms.RandomHorizontalFlip())
    train_trans.append(transforms.ToTensor())
    train_trans = transforms.Compose(train_trans)

    val_trans = []
    val_trans.append(transforms.Resize(size=sample_size))
    val_trans.append(transforms.CenterCrop(size=crop_size))
    val_trans.append(transforms.ToTensor())
    val_trans = transforms.Compose(val_trans)

    # torchvision style datasets
    train_dataset = datasets.ImageFolder(root=train_folder_root, transform=train_trans)
    val_dataset = datasets.ImageFolder(root=val_folder_root, transform=val_trans)

    # dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                                   shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, 
                                                   shuffle=True, num_workers=num_workers)
    return train_dataloader, val_dataloader




    





