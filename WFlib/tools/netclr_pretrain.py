from torch.utils.data.dataset import Dataset
import numpy as np

class PreTrainData(Dataset):
    def __init__(self, x_train, y_train, augmentor, n_views):
        """
        Dataset class for pretraining with data augmentation.

        Parameters:
        x_train (ndarray): Training data.
        y_train (ndarray): Training labels.
        augmentor (Augmentor): Augmentor object to apply augmentations.
        n_views (int): Number of augmented views to generate for each sample.
        """
        self.x = x_train
        self.y = y_train
        self.augmentor = augmentor
        self.n_views = n_views
    
    def _aug(self, inp):
        """
        Apply random swapping augmentation to the input sequence.

        Parameters:
        inp (ndarray): Input sequence.

        Returns:
        ndarray: Augmented sequence.
        """
        flip_idx = np.random.randint(0, 4999, 250)  # Randomly select 250 indices for swapping
        x_w = inp.copy()
        temp = x_w[flip_idx]
        x_w[flip_idx] = x_w[flip_idx + 1]
        x_w[flip_idx + 1] = temp
        return x_w
    
    def __getitem__(self, index):
        """
        Get augmented views and label for a given index.

        Parameters:
        index (int): Index of the sample.

        Returns:
        tuple: List of augmented views and the corresponding label.
        """
        return [self.augmentor.augment(self.x[index]) for _ in range(self.n_views)], self.y[index]
    
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
        int: Number of samples.
        """
        return len(self.x)