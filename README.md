# Automated Pore Detection and Analysis in Materials Using Machine Learning

## Overview

For this project we used machine learning to analyze defects in materials. The machine learning model was used to find and highlight pores in these materials, and from this create masks, that could be used to count the pores in the material and analyze their size distribution across multiple pictures of materials.

## Table of Contents

- [Project Title](#project-title)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of Python 3.11. If not, you can download it from [here](https://www.python.org/downloads/).
- You have a basic understanding of Python programming and machine learning concepts.
- You have installed the necessary Python packages. This project uses several Python packages for data processing, model training, and visualization. You can install them using pip, the Python package installer. Here are the commands to install them:

```bash
pip3 install torch torchvision torchaudio
pip3 install augmentation
pip3 install numpy
pip3 install matplotlib
pip3 install Pillow
pip3 install scikit-learn
pip3 install opencv-python
pip3 install imutils
pip3 install tqdm
pip3 install pathlib
```

## Usage


To train the model, you run the [train.py](src/train.py) in the src folder. Here the important parameters are: LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS, IMAGE_HEIGHT/WIDTH, LOAD_MODEL and the paths for all files. LEARNING_RATE determines, how much the model can change per training run, BATCH_SIZE sets the number of images, that get processed at the same time, NUM_EPOCHS determines the number of training runs. With IMAGE_HEIGHT/WIDTH you can change the size of the used images, and with LOAD_MODEL you can load an already trained model or start from scratch. The path parameters have to point to the place, where your data files are stored and your outputs have to be saved at respectively.
After setting these parameters you run the train.py file.
In the [processing.py](src/processing.py) file you set the path parameters analogous to the train.py file and then run the processing.py file.

To analyze the processed data you run the [analyze.ipynb](src/analyze.ipynb) file. Here the images get converted into boolean arrays, then the holes are counted, saved to a .json file and then plotted. The relevant parameters are again the paths to the files, which have to be set at the end of the respective cells. Also the threshold for the image conversion can be set and the formatting of the plots can be changed.
More information can be found in the docstrings and comments of the files talked about before.

## Dataset

The dataset used was the one provided to us on the Moodle site of the university. This dataset consists of 400 .png images of a material with holes in it. These holes are darker compared to the background, so the model learned to distinguish between the dark patches of the holes and the background.
To prepare the images to use them as training files we used the program Affinity photo to raise the contrast in the images and export them as .jpg and .png files. The same result could be achieved by taking the images and running a threshold filter over each pixel, to get a pure black and white image, where black gets assigned to the holes and white to the background. We used Affinity photo, because that way we could get an immediate visual feedback and see the accuracy of our masks. We created a preset and applied it to every image in the dataset to ensure, that every image was edited with the same parameters.

## Models
In our project, we have implemented a U-Net model to recognize the holes in the material images.

**U-Net:** This model is a type of Convolutional Neural Network that was originally developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg, Germany. The architecture of U-Net is symmetric and consists of two paths: the first one is the contraction path (also known as the encoder) which is used to capture the context in the image. The second one is the expanding path (also known as the decoder) which is used to enable precise localization using transposed convolutions.

The model was trained with a split of 80% training data and 20% test data. We tried various optimizers and loss functions to find the best parameters for the model. The model was trained on a GPU to reduce the training time.

## Evaluation


To evaluate the performance of our U-Net model, we used the Dice Coefficient as our primary metric and visual inspections.

**Metrics:** We used the Dice Coefficient to measure the performance of our model. This metric provides a measure of the overlap between the predicted segmentation and the ground truth, with a higher value indicating better performance.

- **Dice Coefficient: (still work in progress)** This metric calculates the ratio of twice the area of overlap and the sum of the areas of the predicted and actual segmentation. 

**Pixel Accuracy:** We also calculated the pixel accuracy, which is the percentage of pixels that the model correctly identified as either hole or not hole. This gives us a straightforward measure of how often the model is correct.

**Visual Inspection:** In addition to these quantitative metrics, we also performed a visual inspection of the segmentation results. This allowed us to identify any qualitative issues with the model's performance, such as over-segmentation or under-segmentation.

**Validation and Testing Procedures:** We split our dataset into 80% training data and 20% test data. The model was trained on the training data and the metrics were calculated on the test data to evaluate the model's performance. We also used a portion of the training data as a validation set during training, to monitor the model's performance and prevent overfitting.

## Results


We analyzed the training masks and the model masks separately. That way we were able to compare the two methods and visualize the accuracy of the model. It is evident, that the model doesn't perform perfectly, this could be solved by using more datafiles to train the model and perfecting the creation of the training masks.

## Contributing

We welcome contributions of all kinds and are grateful for the time you spend to enhance our project. Here are some guidelines to help you get started:

### Reporting Bugs

If you find a bug, please first check our [issue list](https://github.com/Dawnexa/Automated-Pore-Detection-ML/issues) to see if someone else has already reported it. If not, you can create a new issue, providing as many details as possible about the bug.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing feature, we would love to hear about it. Please create a new issue and describe your idea there. Provide as many details as possible to help us understand and implement your idea.

### Submitting Pull Requests

If you want to contribute code yourself, you can do this by submitting a pull request (PR). Here are the steps you should follow:

1. Fork the repository and create a new branch for your changes.
2. Make your changes in this branch.
3. Submit a PR and describe your changes in the PR form.

Please note that your PR may not be accepted immediately. We may ask you to make some changes before we can accept your PR. We appreciate your patience and your willingness to work with us to improve the project.

## License

This project is licensed under the terms of the MIT license. The full license text is available in the [LICENSE](LICENSE) file.

The MIT license is a permissive license that is short and to the point. It lets people do anything they want with your code as long as they provide attribution back to you and don’t hold you liable.

Please note that this license does not include any datasets or external resources used in the project. These may have their own licenses. Please refer to the respective resources for details.

## Acknowledgements


We would like to express our gratitude to the following individuals and organizations for their contributions, resources, and support:

- [PyTorch](https://pytorch.org) for providing the open-source machine learning library that this project uses.
- [Learn PyTorch](https://www.learnpytorch.io) for the educational resources that helped us understand and implement our models.

We also want to thank all the contributors who have helped with this project, whether by submitting pull requests, reporting bugs, or suggesting improvements. Your help is greatly appreciated.

Please note that this list may not include all the contributions and resources that have helped this project. If you feel that something is missing, please let us know by submitting an issue.