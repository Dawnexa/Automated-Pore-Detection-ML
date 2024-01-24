# import the necessary packages
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_device,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 15
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
PIN_MEMORY = True
LOAD_MODEL = False
BASE_PATH = "project"
TRAIN_IMG_DIR = os.path.sep.join([BASE_PATH, "input", "Dataset", "train_images"])
TRAIN_MASK_DIR = os.path.sep.join([BASE_PATH, "input", "Dataset", "train_masks"])
VAL_IMG_DIR = os.path.sep.join([BASE_PATH, "input", "Dataset", "val_images"])
VAL_MASK_DIR = os.path.sep.join([BASE_PATH, "input", "Dataset", "val_masks"])
SAVE_PRED = os.path.sep.join([BASE_PATH, "output", "saved_images"])
MODEL_PATH = os.path.sep.join([BASE_PATH, "output", "model.pth.tar"])
PLOT_PATH = os.path.sep.join([BASE_PATH, "output", "plot.png"])

def train_fn(loader, model, optimizer, loss_fn, scaler, DEVICE):
    """This function trains the model
    
    Args:
        loader (torch.utils.data.DataLoader): the training data loader
        model (torch.nn.Module): the model to train
        optimizer (torch.optim): the optimizer to use
        loss_fn (torch.nn.Module): the loss function to use
        scaler (torch.cuda.amp.GradScaler): the gradient scaler to use
        
    Returns:
        list: the losses for this epoch
    """
    # Should I write a training message here?
    print(f"[INFO] Starting the training process...")
    loop = tqdm(loader)
    losses = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
          

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        
    return losses

def valid_fn(loader, model, loss_fn, DEVICE):
    """This function validates the model
    
    Args:
        loader (torch.utils.data.DataLoader): the validation data loader
        model (torch.nn.Module): the model to validate
        loss_fn (torch.nn.Module): the loss function to use
    
    Returns:
        list: the losses for this epoch
    """
    # Validation loop message?
    print(f"[INFO] Starting the validation process...")
    model.eval()

    loop = tqdm(loader)
    losses = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.no_grad():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        losses.append(loss.item())

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return losses

def main():
    """
    This function is the main function of the program
    """
    # set the device
    DEVICE = get_device()
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    model = UNet(in_channels=3, out_channels=1).to(DEVICE) # With multi class segmentation we'd do out_channels=3 and change the loss function to nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load("model.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)

    scaler = torch.cuda.amp.GradScaler()

    accuracy = []

    for epoch in range(NUM_EPOCHS):
        print(f"[INFO] Epoch {epoch+1}/{NUM_EPOCHS}")
        train_losses = train_fn(train_loader, model, optimizer, loss_fn, scaler, DEVICE)
        valid_losses = valid_fn(val_loader, model, loss_fn, DEVICE)
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        acc = check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder=SAVE_PRED, device=DEVICE
        )

        print(f"[INFO Current accuracy: {acc:.4f}]")
        accuracy.append(acc.cpu().numpy())
        


    # Now lets plot the accuracy
    print(f"[INFO] Plotting the accuracy...")
    plt.plot(accuracy, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(PLOT_PATH)


if __name__ == "__main__":
    main()
