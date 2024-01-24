import torch.nn as nn
import torch 
import torchvision.transforms.functional as TF 

class DoubleConv(nn.Module):
    """This class represents the double convolutional block used in the UNet architecture
    
    Args:
        in_channels (int): the number of input channels (i.e in a picture with 3 channels (RGB), in_channels would be 3)
        out_channels (int): the number of output channels (i.e in a picture with 1 channel (grayscale), out_channels would be 1)
        
    Returns:
        torch.nn.Module: the double convolutional block
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) # Define the double convolutional block
    def forward(self, x):
        return self.conv(x) # Apply the double convolutional block
    
class UNet(nn.Module):
    """This class represents the UNet architecture
    
    Args:
        in_channels (int): the number of input channels
        out_channels (int): the number of output channels
        features (list): the number of features to use in the different layers (aka the number of output channels for each layer)

    Returns:
        torch.nn.Module: the UNet architecture
    """
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList() # Initialize an empty list for the downsampling part of the UNet
        self.ups = nn.ModuleList() #  Initialize an empty list for the upsampling part of the UNet
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Define the max pooling layer

        # This is basically what happens when we apply max pooling to an image:

            # Suppose we have a 4x4 image (or feature map) and we apply nn.MaxPool2d(kernel_size=2, stride=2). Here's what happens:

            # 1. The image is divided into 2x2 windows. Since both kernel_size and stride are 2, these windows do not overlap.

            # 2. The maximum value is taken from each 2x2 window.

            # 3. These maximum values are used to create the output image. Since each 2x2 window is reduced to a single value, the output image is half the size in height and width.

            # The result is that the spatial size of the image is reduced while the most important features (the maximum values) 
            # are retained. This helps to reduce the number of parameters and prevent overfitting, 
            # while also improving computational efficiency.

        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

            # Up Convolutional Layer, which is sort of the opposite of max pooling.

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """This function performs a forward pass on the UNet architecture
        
        Args:
            x (torch.Tensor): the input tensor
            
        Returns:
            torch.Tensor: the output tensor
        """
        # Initialize an empty list for the skip connections
        skip_connections = []

        # Iterate through the downsampling part of the UNet
        for down in self.downs:
            x = down(x) # Apply the double convolutional block
            skip_connections.append(x) # Add the resulting tensor to the skip connections list
            x = self.pool(x) # Apply max pooling (which halves the size of the tensor)

        x = self.bottleneck(x) # Apply the bottleneck layer
        skip_connections = skip_connections[::-1] # Reverse the skip connections list

        for idx in range(0, len(self.ups), 2): # Iterate through the upsampling part of the UNet
            x = self.ups[idx](x) # Apply the current upsampling layer
            skip_connection = skip_connections[idx//2] # Get the corresponding skip connection

            if x.shape != skip_connection.shape: # If the shapes of the skip connection and the current tensor don't match
                x = TF.resize(x, size=skip_connection.shape[2:]) # Resize the current tensor to match the shape of the skip connection

            concat_skip = torch.cat((skip_connection, x), dim=1) # Concatenate the skip connection and the current tensor
            x = self.ups[idx+1](concat_skip) # Apply the double convolutional block

        return self.final_conv(x) # Apply the final convolutional layer and return the result
    
# def test():
#      x = torch.randn((3, 1, 161, 161))
#      model = UNet(in_channels=1, out_channels=1)
#      preds = model(x)
#      print(preds.shape)
#      print(x.shape)
#      assert preds.shape == x.shape
    
# if __name__ == "__main__":
#      test()
    
    
