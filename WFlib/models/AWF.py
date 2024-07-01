import torch
from torch import nn
import numpy as np

class AWF(nn.Module):
    def __init__(self, num_classes=100, num_tab=1):
        super(AWF, self).__init__()
        
        # Define the feature extraction part of the network using a sequential container
        self.feature_extraction = nn.Sequential(
            nn.Dropout(p=0.25),  # Dropout layer with a 25% dropout rate for regularization

            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # First convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # First max pooling layer

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # Second convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # Second max pooling layer

            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,
                      stride=1, padding='valid', bias=False),  # Third convolutional layer
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.MaxPool1d(kernel_size=4, padding=0),  # Third max pooling layer
        )
        
        # Define the classifier part of the network
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten the tensor to a vector
            nn.Linear(in_features=32*45, out_features=num_classes)  # Fully connected layer for classification
        )

    def forward(self, x):
        # Ensure the input tensor has the expected shape
        assert x.shape[-1] == 3000, f"Expected input with 3000 elements, got {x.shape[-1]}"
        
        # Pass the input through the feature extraction part
        x = self.feature_extraction(x)
        
        # Pass the output through the classifier
        x = self.classifier(x)
        
        return x

if __name__ == '__main__':
    net = AWF(num_classes=100)  # Create an instance of the model with 100 classes
    x = np.random.rand(32, 1, 3000)  # Example input tensor
    x = torch.tensor(x, dtype=torch.float32)  # Convert the input to a torch tensor with float32 type
    out = net(x)  # Perform a forward pass through the network
    print(f"Input shape: {x.shape} --> Output shape: {out.shape}")  # Print the shapes of the input and output tensors
