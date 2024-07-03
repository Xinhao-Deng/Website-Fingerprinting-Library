import os
import sys
import torch
import random
from captum import attr
from tqdm import tqdm
import argparse
import numpy as np
from WFlib import models
from WFlib.tools import data_processor, analyzer
import warnings

# Suppress specific warnings from the 'captum' library
warnings.filterwarnings("ignore", category=UserWarning, module='captum')

# Set a fixed seed for reproducibility of experiments
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="Feature attribution")

# Define command-line arguments
parser.add_argument("--dataset", type=str, required=True, default="DF18", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--device", type=str, default="cpu", help="Device to use, options=[cpu, cuda, cuda:x]")

# Parameters for feature attribution methods
parser.add_argument("--attr_method", type=str, default="DeepLiftShap", 
                    help="Feature attribution method, options=[DeepLiftShap, GradientShap]")

# Threat model parameters
parser.add_argument("--max_num_tabs", type=int, default=1, 
                    help="Maximum number of tabs opened by users while browsing")

# Input parameters
parser.add_argument("--in_file", type=str, default="train", help="Input file name")
parser.add_argument("--feature", type=str, default="DIR", help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

# Optimization parameters
parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for data loader")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of training input data")

# Output parameters
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Directory to save model checkpoints")
parser.add_argument("--save_name", type=str, default="base", help="Name of the model file")

# Parse command-line arguments
args = parser.parse_args()

# Ensure the specified device (e.g., CUDA) is available
if args.device.startswith("cuda"):
    assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
device = torch.device(args.device)

# Construct paths for input dataset and model checkpoints
in_path = os.path.join("./datasets", args.dataset)
ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
out_file = os.path.join(args.checkpoints, args.dataset, args.model, f"attr_{args.attr_method}.npz")

# Check if the dataset path exists
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")

# Check if the output file already exists
if os.path.exists(out_file):
    print(f"{out_file} has been generated.")
    sys.exit(0)

# Load validation data
valid_X, valid_y = data_processor.load_data(os.path.join(in_path, f"{args.in_file}.npz"), args.feature, args.seq_len)
num_classes = len(np.unique(valid_y))

# Ensure there are test samples for all categories
assert num_classes == valid_y.max() + 1, "Labels are not continuous"

# Print dataset information
print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

# Initialize the model
model = eval(f"models.{args.model}")(num_classes, args.max_num_tabs)
model.load_state_dict(torch.load(os.path.join(ckp_path, f"{args.save_name}.pth"), map_location="cpu"))

# Perform feature attribution using the specified method
attr_values = analyzer.feature_attr(model, args.attr_method, valid_X, valid_y, num_classes)

# Print the shape of the attribution values
print("shape of attr_values:", attr_values.shape)

# Save the attribution values to a file
np.savez(out_file, attr_values=attr_values)