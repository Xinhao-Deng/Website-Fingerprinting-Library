import torch.nn as nn
import math
import torch
import numpy as np

class MultiTabRF(nn.Module):
    def __init__(self, num_classes=100):
        """
        Initialize the RF model.

        Parameters:
        num_classes (int): Number of output classes.
        """
        super(MultiTabRF, self).__init__()
        
        # Create feature extraction layers
        features = make_layers([128, 128, 'M', 256, 256, 'M', 512] + [num_classes])
        init_weights = True
        self.first_layer_in_channel = 1
        self.first_layer_out_channel = 32
        
        # Create the initial convolutional layers
        self.first_layer = make_first_layers()
        self.features = features
        self.class_num = num_classes
        
        # Fully connected layer to project to embedding space
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_classes * 65, out_features=num_classes),
        )
        
        # Initialize weights
        if init_weights:
            self._initialize_weights()

    def forward(self, x, num_tabs=1):
        """
        Forward pass of the model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the network.
        """
        x = self.first_layer(x)
        x = x.view(x.size(0), self.first_layer_out_channel, -1)
        x = self.features(x)
        x = self.mlp(x)
        return x

    def _initialize_weights(self):
        """
        Initialize weights for the network layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, in_channels=32):
    """
    Create a sequence of convolutional and pooling layers.

    Parameters:
    cfg (list): Configuration list specifying the layers.
    in_channels (int): Number of input channels.

    Returns:
    nn.Sequential: Sequential container with the layers.
    """
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(3), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]
            in_channels = v

    return nn.Sequential(*layers)

def make_first_layers(in_channels=1, out_channel=32):
    """
    Create the initial convolutional layers.

    Parameters:
    in_channels (int): Number of input channels.
    out_channel (int): Number of output channels.

    Returns:
    nn.Sequential: Sequential container with the initial layers.
    """
    layers = []
    conv2d1 = nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d1, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d2, nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((1, 3)), nn.Dropout(0.1)]

    conv2d3 = nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d3, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    conv2d4 = nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1))
    layers += [conv2d4, nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True), nn.ReLU()]

    layers += [nn.MaxPool2d((2, 2)), nn.Dropout(0.1)]

    return nn.Sequential(*layers)
