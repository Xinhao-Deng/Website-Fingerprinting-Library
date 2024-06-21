from torch.utils.data.dataset import Dataset
import numpy as np

class PreTrainData(Dataset):
    def __init__(self, x_train, y_train, augmentor, n_views):
        self.x = x_train
        self.y = y_train
        self.augmentor = augmentor
        self.n_views = n_views
    
    def _aug(self, inp):
        flip_idx = np.random.randint(0, 4999, 250)
        x_w = inp.copy()
        temp = x_w[flip_idx]
        x_w[flip_idx] = x_w[flip_idx+1]
        x_w[flip_idx+1] = temp
        return x_w
    
    def __getitem__(self, index):
        return [self.augmentor.augment(self.x[index]) for i in range(self.n_views)], self.y[index]
    
    def __len__(self):
        return len(self.x)