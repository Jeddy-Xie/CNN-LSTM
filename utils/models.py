from utils.TSDataset import TimeSeriesDataset
from utils.split_train_val_test import train_test_split, create_dataloaders
from utils.compute_metric import compute_metrics_seq2seq, compute_metrics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time
from torch.utils.data import DataLoader
import gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def odd_kernel(k):
    if k % 2 == 0:
        return k + 1
    else:
        return k

class CNNLSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size,
                 stride,
                 lstm_units,
                 horizon,
                 seq2seq):
        super().__init__()
        self.kernel_size = odd_kernel(kernel_size)
        self.pad = self.kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=0,
        )

        self.lstm = nn.LSTM(filters, lstm_units,
                            num_layers=2, batch_first=True)
        self.seq2seq = seq2seq
        self.horizon  = horizon
        self.fc       = nn.Linear(lstm_units, horizon)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] != self.conv.in_channels:
            x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = F.pad(x, (self.pad, self.pad), mode='reflect')
        
        c = self.conv(x)
        
        c = c.permute(0, 2, 1)
        out, _ = self.lstm(c)

        if self.seq2seq:
            B, W, H = out.shape
            out = out.reshape(B*W, H)
            out = self.fc(out)
            return out.view(B, W, self.horizon)

        else:
            last = out[:, -1, :]
            return self.fc(last)

class WaveNetBlock(nn.Module):
    """
    Single dilated causal convolution block with gated activation, residual and skip connections.
    """
    def __init__(self, in_channels, residual_channels, kernel_size, dilation):
        super().__init__()
        # Padding to ensure causality: pad left = (kernel_size-1)*dilation
        self.pad = (kernel_size - 1) * dilation
        self.conv_filter = nn.Conv1d(
            in_channels, residual_channels,
            kernel_size,
            dilation=dilation,
            padding=self.pad
        )
        self.conv_gate = nn.Conv1d(
            in_channels, residual_channels,
            kernel_size,
            dilation=dilation,
            padding=self.pad
        )
        # 1x1 convolutions for residual and skip pathways
        self.residual_conv = nn.Conv1d(residual_channels, in_channels, 1)
        self.skip_conv     = nn.Conv1d(residual_channels, in_channels, 1)

    def forward(self, x):
        # x: (batch, in_channels, time)
        # Causal convolution: trim the extra padding after conv
        f = torch.tanh(self.conv_filter(x))[:, :, :x.size(2)]
        g = torch.sigmoid(self.conv_gate(x))[:, :, :x.size(2)]
        out = f * g
        # Skip connection
        skip = self.skip_conv(out)
        # Residual connection
        res  = self.residual_conv(out) + x
        return res, skip

class WaveNet(nn.Module):
    """
    WaveNet-style model using stacks of dilated convolution blocks.

    Args:
        in_channels (int): Number of input features.
        residual_channels (int): Channels in the intermediate layers.
        kernel_size (int): Size of the convolution kernel.
        layers (int): Number of layers per stack.
        stacks (int): Number of dilation stacks.
        horizon (int): Number of output time steps (forecast horizon).
    """
    def __init__(self,
                 in_channels: int,
                 residual_channels: int,
                 kernel_size: int,
                 layers: int,
                 stacks: int,
                 horizon: int):
        super().__init__()
        # Initial 1x1 conv to match residual_channels
        self.initial_conv = nn.Conv1d(in_channels, residual_channels, 1)

        # Build dilated stacks
        self.blocks = nn.ModuleList()
        for s in range(stacks):
            for i in range(layers):
                dilation = 2 ** i
                self.blocks.append(
                    WaveNetBlock(residual_channels, residual_channels,
                                 kernel_size, dilation)
                )
        # Post-processing
        self.relu = nn.ReLU()
        self.post_conv1 = nn.Conv1d(residual_channels, residual_channels, 1)
        self.post_conv2 = nn.Conv1d(residual_channels, horizon, 1)

    def forward(self, x):
        # x: (batch, in_channels, time)
        x = self.initial_conv(x)
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        # Sum all skip outputs
        out = sum(skip_connections)
        out = self.relu(out)
        out = self.relu(self.post_conv1(out))
        out = self.post_conv2(out)
        # out shape: (batch, horizon, time)
        return out

# Assume WaveNetBlock and WaveNet are defined or imported above
class WaveNetLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        residual_channels: int,
        kernel_size: int,
        layers: int,
        stacks: int,
        lstm_units: int,
        horizon: int,
        seq2seq: bool = True
    ):
        super().__init__()
        # WaveNet feature extractor
        self.wavenet = WaveNet(
            in_channels=in_channels,
            residual_channels=residual_channels,
            kernel_size=kernel_size,
            layers=layers,
            stacks=stacks,
            horizon=residual_channels   # produce residual_channels features per time-step
        )
        # LSTM on top of WaveNet features
        self.lstm = nn.LSTM(
            input_size=residual_channels,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True
        )
        self.seq2seq = seq2seq
        self.horizon = horizon
        self.fc = nn.Linear(lstm_units, horizon)

    def forward(self, x):
        # x: (B, C, T)
        # WaveNet: returns (B, RF, T) where RF=residual_channels
        feats = self.wavenet(x)  # feats.shape = (B, horizon, T)
        # reshape to (B, T, residual_channels) for LSTM
        feats = feats.permute(0, 2, 1)
        out, _ = self.lstm(feats)  # (B, T, lstm_units)
        if self.seq2seq:
            B, T, H = out.shape
            out = out.reshape(B * T, H)
            out = self.fc(out)  # (B*T, horizon)
            return out.view(B, T, self.horizon)
        else:
            last = out[:, -1, :]  # (B, lstm_units)
            return self.fc(last)  # (B, horizon)