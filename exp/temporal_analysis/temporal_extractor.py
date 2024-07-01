import numpy as np
import os
import argparse
from tqdm import tqdm

def gen_temporal_feature(data, outfile):
    feat_length = 1000
    X = data["X"]
    y = data["y"]

    abs_X = np.absolute(X)
    new_X = []

    for idx in tqdm(range(X.shape[0])):
        temporal_array = np.zeros((2,feat_length))
        loading_time =  abs_X[idx].max()
        interval = 1.0 * loading_time / feat_length

        for packet in X[idx]:
            if packet == 0:
                break
            elif packet > 0:
                order = int(packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[0][order] += 1
            else:
                order = int(-packet / interval)
                if order >= feat_length:
                    order = feat_length - 1
                temporal_array[1][order] += 1
        new_X.append(temporal_array)
    new_X = np.array(new_X)
    print(f"Shape: X = {new_X.shape}, y = {y.shape}")
    np.savez(outfile, X=new_X, y=y)

parser = argparse.ArgumentParser(description='Split datasets')
parser.add_argument("-d", '--dataset', default="Undefended", type=str, help='dataset name')

seed = 2024
args = parser.parse_args()
train_file = f"datasets/{args.dataset}/train.npz"
valid_file = f"datasets/{args.dataset}/valid.npz"

train_data = np.load(train_file)
valid_data = np.load(valid_file)

gen_temporal_feature(train_data, f"datasets/{args.dataset}/temporal_train.npz")
gen_temporal_feature(valid_data, f"datasets/{args.dataset}/temporal_valid.npz")
