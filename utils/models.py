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
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=kernel_size,
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

def cnn_lstm_model_eval(
    df,
    feature_cols,
    target_col,
    scaler,
    filters,
    window_size,
    kernel_size,
    strides,
    lstm_units,
    learning_rate,
    epochs,
    seq2seq=True,
    forecast_horizon=1,
    optimize=True,
    batch_size=64,
    num_workers=4,
    pin_memory=True
):
    """
    End-to-end PyTorch CNN-LSTM training and evaluation supporting both
    single-step (seq2vec) and multi-step (seq2seq) forecasts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Split and build DataLoaders
    train_df, test_df = train_test_split(df, train_frac=0.7)
    train_loader, test_loader, _, _ = create_dataloaders(
        train_df, test_df,
        window_size, forecast_horizon,
        feature_cols, target_col,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seq2seq=seq2seq
    )

    # 2) Model, loss, optimizer
    model = CNNLSTM(
        in_channels=len(feature_cols),
        filters=filters,
        kernel_size=kernel_size,
        stride=strides,
        lstm_units=lstm_units,
        horizon=forecast_horizon,
        seq2seq=seq2seq
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'val_loss': []}

    # 3) Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            X_batch = X_batch.permute(0, 2, 1)
            y_batch = y_batch.permute(0, 2, 1)

            # Check if X_batch ~ (Batch, Features, Windows)
            # Check if y_batch ~ (Batch, Window, Forecast Horizon)
            if X_batch.shape != (int(batch_size), int(len(feature_cols)), int(window_size)):
                print(f"X_batch.shape: {X_batch.shape}")
                print(f"Expected shape: {(int(batch_size), int(len(feature_cols)), int(window_size))}")
                raise ValueError("X_batch shape is incorrect")
            if y_batch.shape != (int(batch_size), int(window_size), int(forecast_horizon)):
                print(f"y_batch.shape: {y_batch.shape}")
                print(f"Expected shape: {(int(batch_size), int(window_size), int(forecast_horizon))}")
                raise ValueError("y_batch shape is incorrect")
            
            # Forward
            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val = X_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)

                X_val = X_val.permute(0, 2, 1)
                y_val = y_val.permute(0, 2, 1)

                # Check if X_val ~ (Batch, Features, Windows)
                # Check if y_val ~ (Batch, Window, Forecast Horizon)
                if X_val.shape != (int(batch_size), int(len(feature_cols)), int(window_size)):
                    print(f"X_val.shape: {X_val.shape}")
                    print(f"Expected shape: {(int(batch_size), int(len(feature_cols)), int(window_size))}")
                    raise ValueError("X_val shape is incorrect")
                if y_val.shape != (int(batch_size), int(window_size), int(forecast_horizon)):
                    print(f"y_val.shape: {y_val.shape}")
                    print(f"Expected shape: {(int(batch_size), int(window_size), int(forecast_horizon))}")
                    raise ValueError("y_val shape is incorrect")

                y_pred = model(X_val)

                val_loss += criterion(y_pred, y_val).item() * X_val.size(0)
        val_loss /= len(test_loader.dataset)
        history['val_loss'].append(val_loss)
        model.train()
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            X_batch = X_batch.permute(0, 2, 1)
            y_batch = y_batch.permute(0, 2, 1)

            # Check if X_batch ~ (Batch, Features, Windows)
            # Check if y_batch ~ (Batch, Window, Forecast Horizon)
            if X_batch.shape != (int(batch_size), int(len(feature_cols)), int(window_size)):
                print(f"X_batch.shape: {X_batch.shape}")
                print(f"Expected shape: {(int(batch_size), int(len(feature_cols)), int(window_size))}")
                raise ValueError("X_batch shape is incorrect")

            if y_batch.shape != (int(batch_size), int(window_size), int(forecast_horizon)):
                print(f"y_batch.shape: {y_batch.shape}")
                print(f"Expected shape: {(int(batch_size), int(window_size), int(forecast_horizon))}")
                raise ValueError("y_batch shape is incorrect")
            
            y_pred = model(X_batch)

            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    # 6) Inverse scale and metrics
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(y_true.shape)

    if seq2seq:
        mse, mae = compute_metrics_seq2seq(y_true_inv, y_pred_inv)
    else:
        mse, mae = compute_metrics(y_true_inv, y_pred_inv)

    return -mse if optimize else (mse, mae, history, model, y_pred_inv, y_true_inv)
