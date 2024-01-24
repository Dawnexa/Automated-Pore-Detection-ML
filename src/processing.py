import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from torchvision.utils import save_image
import numpy as np
import cv2
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_device,
)



BASE_PATH = "project/"
MODEL_PATH = os.path.sep.join([BASE_PATH, "output", "model.pth.tar"]) 
DATA_DIR = os.path.sep.join([BASE_PATH, "input", "Dataset", "train", "images"]) 
RESULT_DIR = os.path.sep.join([BASE_PATH, "output", "results"]) 


def main():
    """This function loads the model and makes predictions on the entire data set"""
    # Get the data that the model should predict
    device = get_device() # get the device
    test_paths = []
    for root, dirs, files in os.walk(DATA_DIR): # walk through the data directory
        for file in files: # for each file in the directory
            if file.endswith(".png"): # if the file is a png file
                test_paths.append(os.path.join(root, file)) # add the path to the file to the list of test paths
    # Load the model 
    model = UNet(in_channels=3, out_channels=1).to(device) # initialize the model
    load_checkpoint(torch.load(MODEL_PATH), model) # load the model from disk
    model.eval() # set the model to evaluation mode
    tensor_dict = {} 

    # Make predictions
    for path in test_paths:
        with torch.no_grad(): # disable gradient calculation
            image = cv2.imread(path) # read the image from disk
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # swap the channels from BGR to RGB (PyTorch expects RGB but OpenCV reads BGR)
            image = image.astype("float32") / 255.0 # scale the pixel values to the range [0, 1]
            image = np.transpose(image, (2, 0, 1)) # transpose the image to the format [channels, height, width]
            image = np.expand_dims(image, axis=0) # add a batch dimension to the image
            image = torch.from_numpy(image).to(device) # convert the image to a tensor and flash it to the device
            prediction = torch.sigmoid(model(image.to(device))) # make the prediction, pass the results through the sigmoid function (to get a probability)
            prediction = torch.squeeze(prediction, dim=0) # remove the batch dimension
            prediction = prediction.cpu().numpy() # convert the prediction to a numpy array
            prediction = prediction > 0.5 # filter out the weak predictions
            filename = os.path.splitext(path.split(os.path.sep)[-1])[0] # get the filename
            prediction_tensor = torch.from_numpy(prediction.astype(np.float32)) # convert the prediction to a tensor
            tensor_dict[filename] = prediction_tensor # add the prediction to the dictionary
            save_image(prediction_tensor.unsqueeze(1), os.path.sep.join([RESULT_DIR, filename + ".png"])) # save the prediction to disk

    print("[INFO] Predictions saved to {}".format(RESULT_DIR))
    torch.save(tensor_dict, os.path.sep.join([RESULT_DIR, "tensor_dict.pt"])) # save the dictionary to disk
                

if __name__ == "__main__":
    main()