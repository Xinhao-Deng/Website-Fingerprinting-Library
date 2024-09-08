import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath, Mlp

# Top-m Multi-Head Self-Attention 
class TopM_MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        """
        Initialize the TopM_MHSA module.

        Parameters:
        embed_dim (int): Dimension of the embedding.
        num_heads (int): Number of attention heads.
        num_mhsa_layers (int): Number of MHSA layers.
        dim_feedforward (int): Dimension of the feedforward layer.
        dropout (float): Dropout rate.
        top_m (int): Number of top elements to keep in attention.
        """
        super().__init__()

        # Create a list of MHSA blocks
        self.nets = nn.ModuleList([
            MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m) 
            for _ in range(num_mhsa_layers)
        ])

    def forward(self, x, pos_embed):
        """
        Forward pass through the TopM_MHSA module.

        Parameters:
        x (Tensor): Input tensor.
        pos_embed (Tensor): Positional embedding.

        Returns:
        Tensor: Output tensor after MHSA blocks.
        """
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output

# Top-m Self-Attention Block
class TopMAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m):
        """
        Initialize the TopMAttention module.

        Parameters:
        dim (int): Dimension of the input.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        top_m (int): Number of top elements to keep in attention.
        """
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        # Linear layer to generate queries, keys, and values
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),  # Softmax for attention scores
            nn.Dropout(dropout),  # Dropout for regularization
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),  # Linear layer for output projection
            nn.Dropout(dropout),  # Dropout for regularization
        )
        self.apply(self._init_weights)  # Initialize weights

    def _init_weights(self, m):
        """
        Initialize the weights of the module.

        Parameters:
        m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass through the TopMAttention module.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after attention.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
        index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x

# Multi-Head Self-Attention Block
class MHSA_Block(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m):
        """
        Initialize the MHSA_Block module.

        Parameters:
        embed_dim (int): Dimension of the embedding.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward layer.
        dropout (float): Dropout rate.
        top_m (int): Number of top elements to keep in attention.
        """
        super().__init__()
        drop_path_rate = 0.1
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)  # Stochastic depth
        self.norm1 = nn.LayerNorm(embed_dim)  # Layer normalization
        self.norm2 = nn.LayerNorm(embed_dim)  # Layer normalization
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)  # Feedforward network

    def forward(self, x):
        """
        Forward pass through the MHSA_Block module.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after MHSA block.
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# Conv1d Block
class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        """
        Initialize the ConvBlock1d module.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        dilation (int): Dilation rate for the convolution.
        """
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the ConvBlock1d module.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after convolutional block.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)

# Local profiling module with several ConvBlock1d
class LocalProfiling(nn.Module):
    def __init__(self):
        """
        Initialize the LocalProfiling module.
        """
        super(LocalProfiling, self).__init__()
        
        self.net = nn.Sequential(
            ConvBlock1d(in_channels=1, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=256, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        """
        Forward pass through the LocalProfiling module.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after local profiling.
        """
        x = self.net(x)
        return x

# ARES model
class ARES(nn.Module):
    def __init__(self, num_classes=100):
        """
        Initialize the ARES model.

        Parameters:
        num_classes (int): Number of output classes.
        """
        super(ARES, self).__init__()
        
        embed_dim = 256
        num_heads = 8
        dim_feedforward = 256 * 4
        num_mhsa_layers = 2
        dropout = 0.1
        max_len = 32
        top_m = 20
        
        # Layer to divide the input into smaller chunks
        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        )
        
        # Layer to combine the chunks back together
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        )
        
        # Local profiling module
        self.profiling = LocalProfiling()
        
        # Positional embedding
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight

        # Top-m Multi-Head Self-Attention module
        self.topm_mhsa = TopM_MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m)
        
        # Fully connected layer for classification
        self.mlp = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the ARES model.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through the model.
        """
        sliding_size = np.random.randint(0, 1 + 2500)
        x = torch.roll(x, shifts=sliding_size, dims=-1)  # Apply random shift for data augmentation
        x = self.dividing(x)
        x = self.profiling(x)
        x = self.combination(x)
        x = x.permute(0, 2, 1)
        x = self.topm_mhsa(x, self.pos_embed.unsqueeze(0))
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x
