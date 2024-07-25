import numpy as np
import pandas as pd
import os
import sys
import random
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

# Set up argument parser to get dataset name from command line arguments
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")

args = parser.parse_args()
in_path = os.path.join("./datasets", f"{args.dataset}")
in_file = os.path.join(in_path, "test.npz")

data = np.load(in_file)
X = data["X"]
y = data["y"]
feat_length = X.shape[1]

for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    out_file = os.path.join(in_path, f"test_p{p}.npz")
    if os.path.exists(out_file):
        continue
    print(f"Generating the page loaded {p}% of traffic")
    cur_X = []
    cur_y = []
    abs_X = np.absolute(X)

    for idx in tqdm(range(X.shape[0])):
        tmp_X = abs_X[idx]
        loading_time = tmp_X.max()
        threshold = loading_time * p / 100
        tmp_X = tmp_X[tmp_X>0]
        tmp_X = tmp_X[tmp_X<=threshold]
        tmp_size = tmp_X.shape[0]

        cur_X.append(np.pad(X[idx][:tmp_size], (0,feat_length-tmp_size), "constant", constant_values=(0,0)))
        cur_y.append(y[idx])
    cur_X = np.array(cur_X)
    cur_y = np.array(cur_y)
    print(f"Shape: X = {cur_X.shape}, y = {cur_y.shape}")
    np.savez(out_file, X = cur_X, y = cur_y)