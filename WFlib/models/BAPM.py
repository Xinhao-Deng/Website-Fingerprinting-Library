import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, output_dim):
        super(AttentionModule, self).__init__()
        self.output_dim = output_dim
        # Initialize weights to None; actual initialization happens in build method
        self.kernel = None

    def build(self, input_shape):
        # This method initializes the weights based on the input shape
        # 3 kernels for Q, K, V respectively
        self.kernel = nn.Parameter(torch.empty(3, input_shape[-1], self.output_dim).cuda())
        nn.init.uniform_(self.kernel)  # Uniform initialization

    def forward(self, x):
        # Check if the kernel has been initialized
        if self.kernel is None:
            self.build(x.shape)
        
        # Compute query, key, and value matrices
        wq, wk, wv = [torch.matmul(x, self.kernel[i]) for i in range(3)]
        # Compute attention scores
        qk = torch.bmm(wq, wk.transpose(1, 2))
        qk = qk / (self.output_dim ** 0.5)
        qk = F.softmax(qk, dim=-1)
        # Apply attention scores to the value matrix
        v = torch.bmm(qk, wv)
        return v


class BAPM(nn.Module):
    def __init__(self, num_classes=100, num_tab=1):
        super(BAPM, self).__init__()

        # Define CNN layer parameters
        filters_num = [32, 64, 128]
        kernels_size = [5, 5, 5]
        pool_size = [8, 8, 8]
        self.time_step = 16
        self.num_tab = num_tab

        # Define CNN layers
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=filters_num[0], kernel_size=kernels_size[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filters_num[0]),
            nn.MaxPool1d(kernel_size=pool_size[0], padding=0),
            nn.Dropout(p=0.2),

            nn.Conv1d(in_channels=filters_num[0], out_channels=filters_num[1], kernel_size=kernels_size[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filters_num[1]),
            nn.MaxPool1d(kernel_size=pool_size[1], padding=0),
            nn.Dropout(p=0.2),

            nn.Conv1d(in_channels=filters_num[1], out_channels=filters_num[2], kernel_size=kernels_size[2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filters_num[2]),
            nn.MaxPool1d(kernel_size=pool_size[2], padding=0),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.2),
        )

        # Define attention heads
        self.one_heads = nn.ModuleList([
            nn.Sequential(
                nn.TransformerEncoderLayer(d_model=128, nhead=1, dim_feedforward=256, batch_first=True),
                nn.Flatten(start_dim=1),
                nn.Dropout(p=0.2),
            ) for _ in range(self.num_tab)
        ])

        # Define fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes),
        )
        
    def forward(self, x):
        # Apply CNN layers
        x = self.cnn_layer(x)
        x = x.view(x.shape[0], self.time_step, -1)

        # Apply attention heads
        multi_head_attn = [self.one_heads[idx](x) for idx in range(self.num_tab)]
        multi_head_attn = [torch.unsqueeze(t, dim=1) for t in multi_head_attn]
        x = torch.cat(multi_head_attn, dim=1)

        # Reshape and apply fully connected layer
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        x_reshaped = self.fc(x_reshaped)
        x = x_reshaped.contiguous().view(x.size(0), -1, x_reshaped.size(-1))
        return x.mean(1)
    