from utils.compute_metric import compute_metrics, compute_metrics_seq2seq
from utils.models import CNNLSTM, WaveNetLSTM
from utils.split_train_val_test import create_dataloaders
import numpy as np
import torch
import gc
from torch.amp import autocast
from torch import nn
import time
from tqdm import tqdm
import pandas as pd

def wavenet_lstm_model_eval(
        
    df,
    npy_path: str,
    data_scaler,
    feat_idx: list[int],
    targ_idx: int,
    dates_npy_path: str | None,
    residual_channels: int,
    kernel_size: int,
    layers: int,
    stacks: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    seq2seq: bool = True,
    forecast_horizon: int = 1,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Train & evaluate WaveNet-LSTM model using memory-mapped dataset.
    Returns -MSE for HPO or full metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build DataLoaders
    train_loader, test_loader, train_ds, test_ds = create_dataloaders(
        df,
        npy_path=npy_path,
        dates_npy_path=dates_npy_path,
        feat_idx=feat_idx,
        targ_idx=targ_idx,
        window_size=train_ds.window_size,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        seq2seq=seq2seq
    )

    # Instantiate model
    model = WaveNetLSTM(
        in_channels=len(feat_idx),
        residual_channels=residual_channels,
        kernel_size=kernel_size,
        layers=layers,
        stacks=stacks,
        lstm_units=lstm_units,
        horizon=forecast_horizon,
        seq2seq=seq2seq
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    # Training loop
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Evaluation
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            preds.append(y_pred.cpu().numpy())
            trues.append(y_batch.cpu().numpy())
    
    y_pred = np.concatenate(preds, 0)
    y_true = np.concatenate(trues, 0)

    # Inverse scale and metrics
    y_pred_inv = data_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(y_pred.shape)
    y_true_inv = data_scaler.inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)
    mse = ((y_pred_inv - y_true_inv)**2).mean()
    mae = np.abs(y_pred_inv - y_true_inv).mean()

    # Cleanup
    del optimizer, criterion, train_loader, test_loader, train_ds, test_ds
    gc.collect()
    torch.cuda.empty_cache()

    return mse, mae, y_pred_inv, y_true_inv