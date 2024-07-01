import torch
import numpy as np

def length_align(X, seq_len):
    """
    Align the length of the sequences to the specified sequence length.
    
    Parameters:
    X (ndarray): Input sequences.
    seq_len (int): Desired sequence length.

    Returns:
    ndarray: Aligned sequences with the specified length.
    """
    if seq_len < X.shape[-1]:
        X = X[...,:seq_len]  # Truncate the sequence if seq_len is shorter than the sequence length
    if seq_len > X.shape[-1]:
        padding_num = seq_len - X.shape[-1]  # Calculate padding length
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)  # Pad the sequence with zeros
    return X

def load_data(data_path, feature_type, seq_len):
    """
    Load and process data from a specified path.

    Parameters:
    data_path (str): Path to the data file.
    feature_type (str): Type of feature to extract.
    seq_len (int): Desired sequence length.

    Returns:
    tuple: Processed feature tensor and label tensor.
    """
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    if feature_type == "DIR":
        X = np.sign(X)  # Directional feature
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "DT":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "DT2":
        X_dir = np.sign(X)
        X_time = np.abs(X)
        X_time = np.diff(X_time)
        X_time[X_time < 0] = 0  # Ensure no negative values
        X_dir = length_align(X_dir, seq_len)[:, np.newaxis]
        X_time = length_align(X_time, seq_len)[:, np.newaxis]
        X = np.concatenate([X_dir, X_time], axis=1)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    elif feature_type == "TAM":
        X = length_align(X, seq_len)
        X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
    else:
        raise ValueError(f"Feature type {feature_type} is not matched.")
    return X, y

def load_iter(X, y, batch_size, is_train=True, num_workers=8):
    """
    Load data into an iterator for batch processing.

    Parameters:
    X (Tensor): Feature tensor.
    y (Tensor): Label tensor.
    batch_size (int): Number of samples per batch.
    is_train (bool): Whether the iterator is for training data.
    num_workers (int): Number of workers for data loading.

    Returns:
    DataLoader: Data loader for batch processing.
    """
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)

def extract_TAM(sequences):
    """
    Extract the Traffic Analysis Matrix (TAM) from sequences.

    Parameters:
    sequences (ndarray): Input sequences.

    Returns:
    ndarray: Extracted TAM features.
    """
    maximum_load_time = 80  # Maximum load time for packets
    max_matrix_len = 1800  # Maximum length of the matrix
    matrix = []
    for idx in range(sequences.shape[0]):
        cur_X = sequences[idx]
        feature = np.zeros((2, max_matrix_len))  # Initialize feature matrix
        for pack in cur_X:
            if pack == 0:
                break  # End of sequence
            elif pack > 0:
                if pack >= maximum_load_time:
                    feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
                else:
                    idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                    feature[0, idx] += 1
            else:
                pack = np.abs(pack)
                if pack >= maximum_load_time:
                    feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
                else:
                    idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                    feature[1, idx] += 1
        matrix.append(feature)

    matrix = np.array(matrix)
    return matrix
