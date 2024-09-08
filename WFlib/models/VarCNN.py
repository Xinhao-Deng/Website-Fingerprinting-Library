import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import weight_norm

class DilatedBasic1D(nn.Module):
    """
    This class defines a basic 1D dilated convolutional block with two convolutional layers,
    batch normalization, ReLU activation, and an optional shortcut connection for residual learning.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilations=(1, 1)):
        super(DilatedBasic1D, self).__init__()
        # First convolutional layer with dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=dilations[0], dilation=dilations[0], bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # Second convolutional layer with dilation
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilations[1], dilation=dilations[1], bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        """
        Defines the forward pass through the block.
        """
        # Apply first convolutional layer, batch norm, and ReLU activation
        out = F.relu(self.bn1(self.conv1(x)))
        # Apply second convolutional layer and batch norm
        out = self.bn2(self.conv2(out))
        # Add the shortcut connection
        out += self.shortcut(x)
        # Apply ReLU activation
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    """
    This class defines an encoder network composed of an initial convolutional block followed by several dilated convolutional blocks.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # Initial convolutional block with padding, convolution, batch norm, ReLU, and max pooling
        self.init_convs = nn.Sequential(*[
                nn.ConstantPad1d(3, 0),
                nn.Conv1d(1, 64, 7, stride=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(3, stride=2, padding=1)
            ])
        # Sequential stack of DilatedBasic1D blocks
        self.convs = nn.Sequential(*[
                DilatedBasic1D(in_channels=64, out_channels=64, stride=1, dilations=[1,2]),
                DilatedBasic1D(in_channels=64, out_channels=64, stride=1, dilations=[4,8]),
                DilatedBasic1D(in_channels=64, out_channels=128, stride=2, dilations=[1,2]),
                DilatedBasic1D(in_channels=128, out_channels=128, stride=1, dilations=[4,8]),
                DilatedBasic1D(in_channels=128, out_channels=256, stride=2, dilations=[1,2]),
                DilatedBasic1D(in_channels=256, out_channels=256, stride=1, dilations=[4,8]),
                DilatedBasic1D(in_channels=256, out_channels=512, stride=2, dilations=[1,2]),
                DilatedBasic1D(in_channels=512, out_channels=512, stride=1, dilations=[4,8]),
            ])
        # Adaptive average pooling to reduce the output to a fixed size
        self.classifier = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Defines the forward pass through the encoder.
        """
        # Pass through initial convolutional block
        x = self.init_convs(x)
        # Pass through dilated convolutional blocks
        x = self.convs(x)
        # Apply adaptive average pooling
        x = self.classifier(x)
        # Flatten the output
        x = x.view(x.shape[0], -1)
        return x

class VarCNN(nn.Module):
    """
    This class defines the overall VarCNN composed of two encoders (directional and temporal)
    and a classifier for final prediction.
    """
    def __init__(self, num_classes):
        super(VarCNN, self).__init__()
        # Two separate encoders for directional and temporal data
        self.dir_encoder = Encoder()
        self.time_encoder = Encoder()
        # Classifier consisting of linear layers, batch norm, ReLU, and dropout
        self.classifier = nn.Sequential(*[
                nn.Linear(in_features=1024, out_features=1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=1024, out_features=num_classes)
            ])
       
    def forward(self, x):
        """
        Defines the forward pass through the VarCNN.
        """
        # Separate input into directional and temporal components and pass through respective encoders
        x_dir = self.dir_encoder(x[:,0:1,:])
        x_time = self.time_encoder(x[:,1:,:])
        # Concatenate the outputs of the two encoders
        x = torch.concat((x_dir,x_time), dim=1)
        # Pass through the classifier
        x = self.classifier(x)
        return x

