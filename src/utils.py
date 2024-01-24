# Import the necessary packages
import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from model import UNet

# Set the Hyperparameters
BASE_PATH = "project"
SAVED_PRED = os.path.sep.join([BASE_PATH,  "output", "saved_images"]) #project/code/output/saved_images
MODEL_PATH = os.path.sep.join([BASE_PATH, "output", "model.pth.tar"]) #project/code/output/model.pth.tar

def save_checkpoint(state, filename=MODEL_PATH): # A model checkpoint is a snapshot of the model parameters and other training state variables at a given time during the training process.
    """This function saves the model checkpoint
    
    Args:
        state (dict): the state of the model
        filename (str, optional): the filename to save the model checkpoint in. Defaults to MODEL_PATH.
    """ 
    # A model checkpoint is a snapshot of the model parameters and other training state variables 
    # at a given time during the training process.

    print("[INFO] saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """This function loads the model checkpoint
    
    Args:
        checkpoint (dict): the state of the model
        model (torch.nn.Module): the model to load the checkpoint into
    """
    print("[INFO] loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    """This function returns the training and validation data loaders
    
    Args:
        train_dir (str): the path to the training images directory
        train_maskdir (str): the path to the training masks directory
        val_dir (str): the path to the validation images directory
        val_maskdir (str): the path to the validation masks directory
        batch_size (int): the batch size to use
        train_transform (torchvision.transforms): the training data transformations to use
        val_transform (torchvision.transforms): the validation data transformations to use
        num_workers (int, optional): the number of workers to use. Defaults to 4.
        pin_memory (bool, optional): whether to pin memory or not. Defaults to True.
        
        Returns:
            tuple: the training and validation data loaders
    """
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory, 
        shuffle=True, # shuffle=True because we want to shuffle the training data
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size, # batch size is the number of samples that will be propagated through the network
        num_workers=num_workers, # num_workers is the number of processes that generate batches in parallel
        pin_memory=pin_memory, # pin_memory is a boolean that decides whether to copy the data into CUDA pinned memory
        shuffle=False, # shuffle is a boolean that decides whether to shuffle the dataset every epoch
    ) # shuffle=False because we don't want to shuffle the validation data

    return train_loader, val_loader

def check_accuracy(loader, model, device="mps"):
    """This function checks the accuracy of the model
    
    Args:
        loader (torch.utils.data.DataLoader): the data loader to use
        model (torch.nn.Module): the model to use
        device (str, optional): the device to use. Defaults to "mps".
        """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
    return num_correct/num_pixels*100 # for some reason the dice score doesn't work as it should

def save_predictions_as_imgs(
    loader, model, folder=SAVED_PRED, device="mps"
):
    """This function saves the predictions as images
    
    Args:
        loader (torch.utils.data.DataLoader): the data loader to use
        model (torch.nn.Module): the model to use
        folder (str, optional): the folder to save the images to. Defaults to "saved_images".
        device (str, optional): the device to use. Defaults to "mps".
    """
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

def get_device():
    """This function finds out what device will be used, and prints the used device out and returns it
    
    Returns:
        torch.device: the device to use
    """
    if not torch.cuda.is_available():
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled, so we will use a CPU instead")
                DEVICE = 'cpu'
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS enabled device on this machine, so we will use a CPU instead")
                DEVICE = 'cpu'
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
    else:
        DEVICE = "cuda"
    print("Using device: {}".format(DEVICE))
    return DEVICE