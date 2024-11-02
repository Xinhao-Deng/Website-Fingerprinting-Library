import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, Mlp

class TopM_MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m):
        super().__init__()
        self.nets = nn.ModuleList([MHSA_Block(embed_dim, num_heads, dim_feedforward, dropout, top_m) for _ in range(num_mhsa_layers)])

    def forward(self, x, pos_embed):
        output = x + pos_embed
        for layer in self.nets:
            output = layer(output)
        return output

class TopMAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m):
        super().__init__()
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim , dim*3)
        self.attn_drop = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout),
        )
        self.proj_drop = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
        index = torch.topk(attn, k=self.top_m, dim=-1, largest=True)[1]
        mask.scatter_(-1, index, 1.)
        attn = torch.where(mask>0, attn, torch.full_like(attn, float('-inf')))

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj_drop(x)
        return x

class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout, top_m):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = TopMAttention(embed_dim, nhead, dropout, top_m)
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels,kernel_size=kernel_size,dilation=dilation, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class LocalProfiling(nn.Module):
    """ Local Profiling module in ARES """
    def __init__(self, in_channels=8):
        super(LocalProfiling, self).__init__()
        
        self.net = nn.Sequential(
            ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
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
        x = self.net(x)
        return x

class ARES(nn.Module):
    def __init__(self, 
                num_classes,
                ):
        super(ARES, self).__init__()

        embed_dim = 256
        num_heads = 8
        dim_feedforward = 256 * 4
        num_mhsa_layers = 4
        dropout = 0.1
        max_len = 29
        top_m=20
        in_channels = 8

        self.profiling = LocalProfiling(in_channels)
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight

        self.topm_mhsa = TopM_MHSA(embed_dim, num_heads, num_mhsa_layers, dim_feedforward, dropout, top_m)
        self.mlp = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.profiling(x)
        x = x.permute(0, 2, 1)
        x = self.topm_mhsa(x, self.pos_embed.unsqueeze(0))
        x = x.mean(dim=1)
        x = self.mlp(x)
        return x

if __name__ == '__main__':
    feat_len = 8000
    in_channels = 8
    net = ARES(num_classes=100)
    # print(net)
    x = np.random.rand(32, in_channels, feat_len)
    x = torch.tensor(x, dtype=torch.float32)
    out = net(x)
    print(f"in:{x.shape} --> out:{out.shape}")