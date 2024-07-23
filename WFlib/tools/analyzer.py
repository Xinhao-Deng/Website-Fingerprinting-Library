from captum import attr
from tqdm import tqdm
import torch
import numpy as np

def feature_attr(model, attr_method, X, y, num_classes):
    """
    Calculate feature attributions for a given model using a specified attribution method.
    
    Args:
    - model: The neural network model to interpret.
    - attr_method: The attribution method to use (e.g., 'DeepLiftShap').
    - X: The input data (features) as a numpy array or torch tensor.
    - y: The labels for the input data.
    - num_classes: The number of distinct classes in the data.
    
    Returns:
    - attr_values: An array of attribution values for each class.
    """
    
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize the attribution model based on the chosen method
    if attr_method in ["DeepLiftShap"]:
        attr_model = eval(f"attr.{attr_method}")(model)
    else:
        attr_model = eval(f"attr.{attr_method}")(model.forward)
    
    # Prepare background and test data for each class
    bg_traffic = []
    test_traffic = {}
    for web in range(num_classes):
        bg_test_X = X[y == web]
        assert bg_test_X.shape[0] >= 12
        bg_traffic.append(bg_test_X[0:2])  # Use the first 2 samples as background
        test_traffic[web] = bg_test_X[2:12]  # Use the next 10 samples for testing

    # Concatenate all background traffic into a single tensor
    bg_traffic = torch.concat(bg_traffic, axis=0)

    attr_values = []
    # Iterate over each class to calculate attribution values
    for web in tqdm(range(num_classes)):
        # Calculate attributions for the test samples using the background samples
        attr_result = attr_model.attribute(test_traffic[web], bg_traffic, target=web)
        # Aggregate the attribution results
        attr_result = attr_result.detach().numpy().squeeze().sum(axis=0).sum(axis=0)
        attr_values.append(attr_result)
    
    attr_values = np.array(attr_values)
    return attr_values  # Return the attribution values
