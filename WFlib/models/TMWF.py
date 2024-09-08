import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        """
        Transformer model with encoder and decoder.

        Parameters:
        embed_dim (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        """
        super().__init__()

        self.encoder = TransformerEncoder(TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        """
        Forward pass of the Transformer model.

        Parameters:
        src (Tensor): Input source sequence.
        query_embed (Tensor): Query embeddings.
        pos_embed (Tensor): Positional embeddings.

        Returns:
        Tensor: Output of the decoder.
        """
        bs = src.shape[0]
        memory = self.encoder(src, pos_embed)
        query_embed = query_embed.repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)
        output = self.decoder(tgt, memory, pos_embed, query_embed)

        return output

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        """
        Transformer encoder composed of multiple encoder layers.

        Parameters:
        encoder_layer (nn.Module): Encoder layer.
        num_layers (int): Number of encoder layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos):
        """
        Forward pass of the Transformer encoder.

        Parameters:
        src (Tensor): Input source sequence.
        pos (Tensor): Positional embeddings.

        Returns:
        Tensor: Encoder output.
        """
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        """
        Transformer decoder composed of multiple decoder layers.

        Parameters:
        decoder_layer (nn.Module): Decoder layer.
        num_layers (int): Number of decoder layers.
        """
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos):
        """
        Forward pass of the Transformer decoder.

        Parameters:
        tgt (Tensor): Target sequence.
        memory (Tensor): Encoder memory.
        pos (Tensor): Positional embeddings.
        query_pos (Tensor): Query positional embeddings.

        Returns:
        Tensor: Decoder output.
        """
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        """
        Transformer encoder layer.

        Parameters:
        embed_dim (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, pos):
        """
        Forward pass of the Transformer encoder layer.

        Parameters:
        src (Tensor): Input source sequence.
        pos (Tensor): Positional embeddings.

        Returns:
        Tensor: Encoder layer output.
        """
        src2 = self.self_attn(query=src + pos, key=src + pos, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        """
        Transformer decoder layer.

        Parameters:
        embed_dim (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float): Dropout rate.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, pos, query_pos):
        """
        Forward pass of the Transformer decoder layer.

        Parameters:
        tgt (Tensor): Target sequence.
        memory (Tensor): Encoder memory.
        pos (Tensor): Positional embeddings.
        query_pos (Tensor): Query positional embeddings.

        Returns:
        Tensor: Decoder layer output.
        """
        tgt2 = self.self_attn(query=tgt + query_pos, key=tgt + query_pos, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class DFNet(nn.Module):
    def __init__(self, dropout):
        """
        Deep Fingerprinting Network (DFNet).

        Parameters:
        dropout (float): Dropout rate.
        """
        super(DFNet, self).__init__()

        filter_num = [0, 32, 64, 128, 256]
        kernel_size = [0, 8, 8, 8, 8]
        conv_stride_size = [0, 1, 1, 1, 1]
        pool_stride_size = [0, 4, 4, 4, 4]
        pool_size = [0, 8, 8, 8, 8]

        self.block1_conv1 = nn.Conv1d(in_channels=1, out_channels=filter_num[1],
                                      kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn1 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn2 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=pool_size[1] // 2)
        self.block1_dropout = nn.Dropout(p=dropout)

        self.block2_conv1 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn1 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn2 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=pool_size[2] // 2)
        self.block2_dropout = nn.Dropout(p=dropout)

        self.block3_conv1 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn1 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn2 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=pool_size[3] // 2)
        self.block3_dropout = nn.Dropout(p=dropout)

        self.block4_conv1 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn1 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn2 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=pool_size[4] // 2)
        self.block4_dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        """
        Forward pass of the DFNet.

        Parameters:
        input (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after passing through all blocks.
        """
        if len(input.shape) == 2:
            x = input.unsqueeze(1)
        else:
            x = input

        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_relu2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_pool(x)
        x = self.block4_dropout(x)
        return x.transpose(1, 2)

class BasicBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Basic block for residual learning.

        Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        """
        super(BasicBlock, self).__init__()
        self._norm_layer = torch.nn.BatchNorm1d
        self.stride = 1
        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels, 3, self.stride, padding=1),
            self._norm_layer(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv1d(out_channels, out_channels, 3, padding=1),
            self._norm_layer(out_channels),
            torch.nn.ReLU(),
        )

        if in_channels != out_channels:
            self.res_layer = torch.nn.Conv1d(in_channels, out_channels, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Parameters:
        x (Tensor): Input tensor.

        Returns:
        Tensor: Output tensor after residual learning.
        """
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual

class TMWF_noDF(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, max_len, num_queries,
                 cls, dropout):
        """
        Transformer Model with CNN without DFNet (TMWF_noDF).

        Parameters:
        embed_dim (int): Dimension of the embeddings.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        num_encoder_layers (int): Number of encoder layers.
        num_decoder_layers (int): Number of decoder layers.
        max_len (int): Maximum length of the sequence.
        num_queries (int): Number of queries.
        cls (int): Number of output classes.
        dropout (float): Dropout rate.
        """
        super(TMWF_noDF, self).__init__()
        print('TMWF_noDF')
        self.cnn_layer = BAPM_CNN([32, 64, 128], [5, 5, 5], [8, 8, 8], dropout)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.trm = Transformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.query_embed = nn.Embedding(num_queries, embed_dim).weight
        self.fc = nn.Linear(embed_dim, cls)

    def forward(self, input):
        """
        Forward pass of the TMWF_noDF model.

        Parameters:
        input (Tensor): Input tensor.

        Returns:
        Tensor: Logits after passing through the model.
        """
        x = self.cnn_layer(input)
        feat = self.proj(x)
        o = self.trm(feat, self.query_embed.unsqueeze(0), self.pos_embed.unsqueeze(0))
        logits = self.fc(o)

        return logits

class TMWF(nn.Module):
    def __init__(self, num_classes=100, num_tab=1):
        """
        Transformer Model with CNN and DFNet (TMWF).

        Parameters:
        num_classes (int): Number of output classes.
        num_tab (int): Number of tabs.
        """
        super(TMWF, self).__init__()
        embed_dim = 256
        nhead = 8
        dim_feedforward = 256 * 4
        num_encoder_layers = 2
        num_decoder_layers = 2
        max_len = 121
        num_queries = num_tab
        dropout = 0.1

        self.cnn_layer = DFNet(dropout)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.trm = Transformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.query_embed = nn.Embedding(num_queries, embed_dim).weight
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, input):
        """
        Forward pass of the TMWF model.

        Parameters:
        input (Tensor): Input tensor.

        Returns:
        Tensor: Logits after passing through the model.
        """
        x = self.cnn_layer(input)
        feat = self.proj(x)
        o = self.trm(feat, self.query_embed.unsqueeze(0), self.pos_embed.unsqueeze(0))
        logits = self.fc(o)
        
        return logits.mean(1)
