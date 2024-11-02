import os
import sys
import json
import random
import torch
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from WFlib import models
from WFlib.tools import data_processor, evaluator

# Set a fixed seed for reproducibility of experiments
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="Spatial analysis")

# Define command-line arguments
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--device", type=str, default="cpu", help="Device to use, options=[cpu, cuda, cuda:x]")

# Input parameters
parser.add_argument("--valid_file", type=str, default="valid", help="Validation file name")
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
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")
ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
out_file = os.path.join(ckp_path, "spatial_distribution.npz")

# Check if the output file already exists
if os.path.exists(out_file):
    print(f"{out_file} has been generated.")
    sys.exit(0)

# Load validation and test data
valid_X, valid_y = data_processor.load_data(os.path.join(in_path, f"{args.valid_file}.npz"), args.feature, args.seq_len)
num_classes = len(np.unique(valid_y))

# Ensure there are test samples for all categories
assert num_classes == valid_y.max() + 1, "Labels are not continuous"

# Print dataset information
print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

# Load data into iterators
valid_iter = data_processor.load_iter(valid_X, valid_y, args.batch_size, False, args.num_workers)

# Initialize the model
model = eval(f"models.{args.model}")(num_classes)
model.load_state_dict(torch.load(os.path.join(ckp_path, f"{args.save_name}.pth"), map_location="cpu"))
model.to(device)

# Initialize a dictionary to store embeddings for each class
embs_pool = {}
for web in range(num_classes):
    embs_pool[web] = []

# Collect embeddings for the validation set
with torch.no_grad():
    model.eval()
    for index, cur_data in enumerate(valid_iter):
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
        embs = model(cur_X).cpu().numpy()
        for i, web in enumerate(cur_y.cpu().numpy()):
            embs_pool[web].append(embs[i])

# Calculate centroids and radii for each class
webs_centroid = []
webs_radius = []
for web in range(num_classes):
    cur_embs = np.array(embs_pool[web])
    cur_centroid = cur_embs.mean(axis=0)
    webs_centroid.append(cur_centroid) 

    cur_radius = 1.0 - cosine_similarity(cur_embs, cur_centroid.reshape(1, -1))
    webs_radius.append(evaluator.median_absolute_deviation(cur_radius))

webs_centroid = np.array(webs_centroid)
webs_radius = np.array(webs_radius)

# Adjust radii to ensure no overlap between different classes
for web1 in range(num_classes):
    for web2 in range(web1 + 1, num_classes):
        centroid_1 = webs_centroid[web1]
        centroid_2 = webs_centroid[web2]
        distance = 1.0 - cosine_similarity(centroid_1.reshape(1, -1), centroid_2.reshape(1, -1))[0, 0]
        radius_1 = webs_radius[web1]
        radius_2 = webs_radius[web2]
        if distance <= radius_1 + radius_2:
            print(f"{web1} vs {web2}: distance = {distance}, r1 = {webs_radius[web1]}, r2 = {webs_radius[web2]}")
            diff = radius_1 + radius_2 - distance
            webs_radius[web1] -= 1.0 * diff * radius_1 / (radius_1 + radius_2)
            webs_radius[web2] -= 1.0 * diff * radius_2 / (radius_1 + radius_2)

# Print completion message and save results
print(f"Generate {out_file} done")
np.savez(out_file, centroid=webs_centroid, radius=webs_radius)
