# import the necessary packages
import numpy as np 
import matplotlib.pyplot as plt
import torch 
import cv2
import os 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import model
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    get_device
)

# Hyperparameters etc.
BASE_PATH = "project"


INPUT_IMAGE_WIDTH = 64
INPUT_IMAGE_HEIGHT = 64


TEST_IMAGE_PATH = os.path.sep.join([BASE_PATH, "output", "test_paths.txt"]) 
IMAGES_PATH = os.path.sep.join([BASE_PATH, "input", "Dataset", "val_images"]) 
MASKS_PATH = os.path.sep.join([BASE_PATH, "input", "Dataset", "val_masks"]) 
MODEL_PATH = os.path.sep.join([BASE_PATH, "output", "model.pth.tar"]) 
SAVE_PRED = os.path.sep.join([BASE_PATH,  "output", "preds"]) 

# grab the testing image paths and sort them
testImages = sorted(list(paths.list_images(IMAGES_PATH)))
testMasks = sorted(list(paths.list_images(MASKS_PATH)))


# write the testing image paths to disk so that we can use them when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(TEST_IMAGE_PATH, "w")
f.write("\n".join(testImages))
f.close()

def prepare_plot(origImage, origMask, predMask, filename):
    """This function prepares a plot of the original image, its mask and the predicted mask
    and saves it to disk
    Args:
        origImage (numpy.ndarray): the original image
        origMask (numpy.ndarray): the original mask
        predMask (numpy.ndarray): the predicted mask
        filename (str): the filename to save the plot in
    """
    filename = os.path.sep.join([SAVE_PRED, filename + ".png"])
    # initialize our figure 
    figure, ax = plt.subplots(1, 3, figsize=(10, 10))

    # plot the original image, its mask and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)

    # set the titles for each image
    ax[0].set_title("Original Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")

    # set the layout of the figure and display it 
    figure.tight_layout()
    figure.savefig(filename)

def make_prediction(model, imagePath, filename, device):
    """This function makes a prediction on a single image
    Args:
        model (torch.nn.Module): the model to be used for prediction
        imagePath (str): the path to the image to be used for prediction
        filename (str): the filename to save the plot in
    """
    # set model to evaluation mode 
    model.eval()

    with torch.no_grad():
        # load the image from disk, swap its channels from BGR to RGB, cast it 
        # to float and scale its pixel values 

        image = cv2.imread(imagePath) # read the image from disk
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # swap the channels from BGR to RGB (PyTorch expects RGB but OpenCV reads BGR)
        image = image.astype("float32") / 255.0 # scale the pixel values to the range [0, 1]

        # resize the image and make a copy of it for visualization purposes
        image = cv2.resize(image, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))
        orig = image.copy()

        # find the filename and generate the path to ground truth
        filename = os.path.splitext(imagePath.split(os.path.sep)[-1])[0]
        maskPath = os.path.sep.join([MASKS_PATH, filename + ".jpg"])

        # load the ground truth mask in grayscale and resize it 
        mask = cv2.imread(maskPath, 0)
        mask = cv2.resize(mask, (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT))

        # make the channel axis to be the leading one, add a batch 
        # dimension to the image and mask, and convert them to tensors
        # and flash it to the device
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image).to(device)

        # make the prediction, pass the results through the sigmoid function 
        # and convert them to a numpy array
        pred = model(image).squeeze()
        pred = torch.sigmoid(pred)
        pred = (pred).cpu().numpy()

        # filter out the weak predictions and convert them to integers
        pred = (pred > 0.5) * 255
        pred = pred.astype(np.uint8)

        # prepare the plot
        prepare_plot(orig, mask, pred, filename)


def main():
    """This is the main function of the script"""

    device = get_device() # get the device to use
    # load the image paths in our testing file and randomly select 10 of them
    print("[INFO] loading testing image paths...") # load the testing image paths
    imagePaths = open(TEST_IMAGE_PATH).read().strip().split("\n") # load the image paths
    imagePaths = np.random.choice(imagePaths, size=10) # randomly select 10 of them

    # load the model from disk
    print("[INFO] loading model from disk...")
    model = model.UNet().to(device)
    load_checkpoint(torch.load(MODEL_PATH), model) # load the model from disk

    # loop over the image paths
    for imagePath in imagePaths:
        # generate the path for the plot to be saved in
        filename = os.path.splitext(imagePath.split(os.path.sep)[-1])[0]
        # make a prediction on the image and display it
        make_prediction(model, imagePath, filename, device=device)

if __name__ == "__main__":
    main() 