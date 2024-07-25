# Generates the traffic aggregation features. More details can be found in the paper: 
# Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis. CCS 2024.
import numpy as np
import os
import argparse
from typing import List
import time
import random
import torch
from tqdm import tqdm
from multiprocessing import Process
from WFlib.tools import data_processor

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
parser.add_argument("--in_file", type=str, default="train", help="input file")

# Parse arguments
args = parser.parse_args()
in_path = os.path.join("./datasets", args.dataset)
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")

# Define output file path
out_file = os.path.join(in_path, f"taf_{args.in_file}.npz")

# If the output file does not exist, process the input file
if not os.path.exists(out_file):
    # Load dataset from the specified .npz file
    data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))
    X = data["X"]
    y = data["y"]
    # Align the sequence length
    X = data_processor.length_align(X, args.seq_len)
    # Extract the TAF
    X = data_processor.extract_TAF(X)
    # Print processing information
    print(f"{args.in_file} process done: X = {X.shape}, y = {y.shape}")
    # Save the processed data into a new .npz file
    np.savez(out_file, X = X, y = y)
else:
    # Print a message if the output file already exists
    print(f"{out_file} has been generated.")