from captum import attr
from tqdm import tqdm
import torch
import numpy as np

def feature_attr(model, attr_method, X, y, num_classes):
    model.eval()
    
    if attr_method in ["DeepLiftShap"]:
        attr_model = eval(f"attr.{attr_method}")(model)
    else:
        attr_model = eval(f"attr.{attr_method}")(model.forward)
    
    bg_traffic = []
    test_traffic = {}
    for web in range(num_classes):
        bg_test_X = X[y == web]
        assert bg_test_X.shape[0] >= 12
        bg_traffic.append(bg_test_X[0:2])
        test_traffic[web] = bg_test_X[2:12]
    bg_traffic = torch.concat(bg_traffic, axis=0)


    attr_values = []
    for web in tqdm(range(num_classes)):
        attr_result = attr_model.attribute(test_traffic[web], bg_traffic, target=web)
        attr_result = attr_result.detach().numpy().squeeze().sum(axis=0).sum(axis=0)
        attr_values.append(attr_result)
    
    attr_values = np.array(attr_values)
    return attr_values
        
