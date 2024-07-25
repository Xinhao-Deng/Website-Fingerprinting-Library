# Offline data augmentation method of Holmes.
# Details can be found in https://arxiv.org/pdf/2407.00918
import os
import argparse
import numpy as np
from tqdm import tqdm
import random

def gen_augment(data, num_aug, effective_ranges, out_file):
    """
    Generate augmented data based on the provided dataset and save it to a file.
    
    Parameters:
    data (dict): Dictionary containing 'X' (features) and 'y' (labels) from the dataset.
    num_aug (int): Number of augmentations to generate per original sample.
    effective_ranges (dict): Dictionary specifying the effective ranges for each class.
    out_file (str): Path to the output file to save the augmented data.
    """
    X = data["X"]
    y = data["y"]

    new_X = []
    new_y = []
    abs_X = np.absolute(X)
    feat_length = X.shape[1]

    # Loop through each sample in the dataset
    for index in tqdm(range(abs_X.shape[0])):
        cur_abs_X = abs_X[index]
        cur_web = y[index]
        loading_time = cur_abs_X.max()

        # Generate augmentations for each sample
        for ii in range(num_aug):
            p = np.random.randint(effective_ranges[cur_web][0], effective_ranges[cur_web][1])
            threshold = loading_time * p / 100
            valid_X = cur_abs_X[cur_abs_X > 0]
            valid_X = valid_X[valid_X <= threshold]
            valid_length = valid_X.shape[0]
            new_X.append(np.pad(X[index][:valid_length], (0, feat_length - valid_length), "constant", constant_values=(0, 0)))
            new_y.append(cur_web)

        # Add the original sample
        new_X.append(X[index])
        new_y.append(cur_web)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    # Save the augmented data to the specified output file
    np.savez(out_file, X=new_X, y=new_y)
    print(f"Generate {out_file} done.")

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")

# Define command-line arguments
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--in_file", type=str, default="train", help="Input file name")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Directory to save model checkpoints")
parser.add_argument("--attr_method", type=str, default="DeepLiftShap", 
                    help="Feature attribution method, options=[DeepLiftShap, GradientShap]")

# Parse command-line arguments
args = parser.parse_args()

# Construct the input path for the dataset
in_path = os.path.join("./datasets", args.dataset)
data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))

# Load the temporal attribution data
temporal_data = np.load(os.path.join(args.checkpoints, args.dataset, args.model, f"attr_{args.attr_method}.npz"))["attr_values"]

# Calculate effective ranges for each class based on the temporal attribution data
effective_ranges = {}
for web in range(temporal_data.shape[0]):
    cur_temporal = np.cumsum(temporal_data[web])
    cur_temporal /= cur_temporal.max()
    cur_lower = np.searchsorted(cur_temporal, 0.3, side="right") * 100 // temporal_data.shape[1]
    cur_upper = np.searchsorted(cur_temporal, 0.6, side="right") * 100 // temporal_data.shape[1]
    effective_ranges[web] = (cur_lower, cur_upper)

# Construct the output file path for the augmented data
out_file = os.path.join(in_path, f"aug_{args.in_file}.npz")

# Check if the output file already exists
if not os.path.exists(out_file):
    # Generate augmented data and save it to the output file
    gen_augment(data, 2, effective_ranges, out_file)
else:
    # If the output file already exists, print a message indicating it has been generated
    print(f"{out_file} has been generated.")