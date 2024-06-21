import torch
import numpy as np

def length_align(X, seq_len):
    if seq_len < X.shape[-1]:
        X = X[...,:seq_len]
    if seq_len > X.shape[-1]:
        padding_num = seq_len - X.shape[-1]
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)
    return X

def load_data(data_path, feature_type, seq_len):
    data = np.load(data_path)
    X = data["X"]
    y = data["y"]

    if feature_type == "DIR":
        X = np.sign(X)
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
        X_time[X_time<0] = 0
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

def load_iter(X, y, batch_size, is_train = True, num_workers=8):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)

def extract_TAM(sequences):
    maximum_load_time = 80
    max_matrix_len = 1800
    matrix = []
    for idx in range(sequences.shape[0]):
        cur_X = sequences[idx]
        feature = np.zeros((2, max_matrix_len))
        for pack in cur_X:
            if pack == 0:
                break
            elif pack > 0:
                if pack >= maximum_load_time:
                    feature[0,-1] += 1
                else:
                    idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                    feature[0,idx] += 1
            else:
                pack = np.abs(pack)
                if pack >= maximum_load_time:
                    feature[1,-1] += 1
                else:
                    idx = int(pack * (max_matrix_len - 1) / maximum_load_time)
                    feature[1,idx] += 1
        matrix.append(feature)

    matrix = np.array(matrix)
    return matrix