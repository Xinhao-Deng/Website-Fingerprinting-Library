import os
import json
import time
import torch
import random
import argparse
import numpy as np
from WFlib import models
from WFlib.tools import data_processor, model_utils, netclr_pretrain

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--device", type=str, default="cpu", help="Device, options=[cpu, cuda, cuda:x]")

# Threat model parameters
parser.add_argument("--max_num_tabs", type=int, default=1, 
                    help="Maximum number of tabs opened by users while browsing")

# Input parameters
parser.add_argument("--train_file", type=str, default="train", help="Train file")
parser.add_argument("--valid_file", type=str, default="valid", help="Valid file")
parser.add_argument("--feature", type=str, default="DIR", 
                    help="Feature type, options=[DIR, DT, DT2, TAM, TAF]")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")

# Optimization parameters
parser.add_argument("--num_workers", type=int, default=10, help="Data loader num workers")
parser.add_argument("--train_epochs", type=int, default=30, help="Train epochs")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size of train input data")
parser.add_argument("--learning_rate", type=float, default=2e-3, help="Optimizer learning rate")
parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer")
parser.add_argument("--loss", type=str, default="CrossEntropyLoss", help="Loss function")
parser.add_argument("--lradj", type=str, default="None", 
                    help="adjust learning rate, option=[None, StepLR]")

# Data augmentation parameters
parser.add_argument("--augmentation_ratio", type=int, default=0, help="How many times to augment")
parser.add_argument("--augmentation_type", type=str, default="offline", help="Type of data augmentation")
parser.add_argument("--augmentation_method", type=str, default="NetAug", help="Method of data augmentation")

# Output
parser.add_argument('--eval_metrics', nargs='+', required=True, type=str, 
                    help="Evaluation metrics, options=[Accuracy, Precision, Recall, F1-score, P@min, r-Precision]")
parser.add_argument("--save_metric", type=str, default="F1-score", 
                    help="Save the model when the metric reaches its maximum value on the validation set")
parser.add_argument("--log_path", type=str, default="./logs/", help="Log path")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Location of model checkpoints")
parser.add_argument("--save_name", type=str, default="base", help="Name of the model file")

# Parse arguments
args = parser.parse_args()

if args.device.startswith("cuda"):
    assert torch.cuda.is_available(), f"The specified device {args.device} does not exist"
device = torch.device(args.device)

# Define paths for dataset, logs, and checkpoints
in_path = os.path.join("./datasets", args.dataset)
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")
log_path = os.path.join(args.log_path, args.dataset, args.model)
ckp_path = os.path.join(args.checkpoints, args.dataset, args.model)
os.makedirs(log_path, exist_ok=True)
os.makedirs(ckp_path, exist_ok=True)

# Load training and validation data
train_X, train_y = data_processor.load_data(os.path.join(in_path, f"{args.train_file}.npz"), args.feature, args.seq_len)
valid_X, valid_y = data_processor.load_data(os.path.join(in_path, f"{args.valid_file}.npz"), args.feature, args.seq_len)
num_classes = len(np.unique(train_y))

# Ensure labels are continuous
assert num_classes == train_y.max() + 1, "Labels are not continuous"

# Print dataset information
print(f"Train: X={train_X.shape}, y={train_y.shape}")
print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

# Load data into iterators
train_iter = data_processor.load_iter(train_X, train_y, args.batch_size, True, args.num_workers)
valid_iter = data_processor.load_iter(valid_X, valid_y, args.batch_size, False, args.num_workers)

# Initialize model, optimizer, and loss function
model = eval(f"models.{args.model}")(num_classes, args.max_num_tabs)
optimizer = eval(f"torch.optim.{args.optimizer}")(model.parameters(), lr=args.learning_rate)

if args.model == "NetCLR": # Pre-training for NetCLR
    pretrain_model = eval(f"models.{args.model}")(num_classes, args.max_num_tabs)
    pretrain_model.update_fc(out_dim=128)
    pretrain_model.cuda()
    random_X = train_X[np.random.choice(range(len(train_X)), size=1000, replace=False)]
    augmentor = model_utils.build_augmentor(random_X)
    pretrain_dataset = netclr_pretrain.PreTrainData(train_X.numpy().squeeze(1), train_y.numpy(), augmentor, 2)
    pretrain_iter = torch.utils.data.DataLoader(pretrain_dataset, args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    pretrain_file = os.path.join(ckp_path, f"pretrain.pth")
    model_utils.model_pretrian(
        pretrain_model,
        optimizer,
        pretrain_iter,
        100,
        pretrain_file,
        args.batch_size
    )

    checkpoint = torch.load(pretrain_file)
    for k in list(checkpoint.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                checkpoint[k[len("backbone."):]] = checkpoint[k]
        del checkpoint[k]
    log = model.load_state_dict(checkpoint, strict=False)
    train_iter = valid_iter

model.cuda()
model_utils.model_train(
    model, 
    optimizer, 
    train_iter, 
    valid_iter, 
    args.loss, 
    args.save_metric, 
    args.eval_metrics, 
    args.train_epochs,
    os.path.join(ckp_path, f"{args.save_name}.pth"),
    num_classes
)