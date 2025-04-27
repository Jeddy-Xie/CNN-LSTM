from utils.compute_metric import compute_metrics, compute_metrics_seq2seq
from utils.models import CNNLSTM, WaveNetLSTM
from utils.split_train_val_test import create_dataloaders
import numpy as np
import torch
import gc
from torch.amp import autocast, GradScaler
from torch import nn
import time
from tqdm import tqdm
import pandas as pd

def cnn_lstm_model_eval(
    df: pd.DataFrame,
    npy_path: str,
    data_scaler,
    feat_idx: list[int],
    targ_idx: int,
    filters: int,
    window_size: int,
    kernel_size: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    forecast_horizon: int,
    batch_size: int = 64,
    seq2seq: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    strides: int = 1,
):
    """
    Final CNN-LSTM training & evaluation on the full dataset.

    Trains the model for `epochs` over the entire train split, then
    runs a single pass over the test split to produce predictions.
    Prints per-epoch training times and overall test time.

    Returns:
        mse (float), mae (float),
        history (dict of train_loss),
        model (trained nn.Module),
        y_pred_inv (np.ndarray),
        y_true_inv (np.ndarray)
    """
    # Device + AMP setup
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    scaler   = torch.amp.GradScaler(enabled=use_cuda)
    amp_args = dict(device_type='cuda', enabled=use_cuda)

    # 1) DataLoaders
    train_loader, test_loader, train_ds, test_ds = create_dataloaders(
        df,
        npy_path=npy_path,
        dates_npy_path=None,
        feat_idx=feat_idx,
        targ_idx=targ_idx,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        num_workers=(num_workers if use_cuda else 0),
        pin_memory=(pin_memory and use_cuda),
        persistent_workers=(use_cuda and num_workers > 0),
        seq2seq=seq2seq
    )

    # 2) Model, loss, optimizer
    model = CNNLSTM(
        in_channels=len(feat_idx),
        filters=filters,
        kernel_size=kernel_size,
        stride=strides,
        lstm_units=lstm_units,
        horizon=forecast_horizon,
        seq2seq=seq2seq
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': []}

    # 3) Training
    print("Starting training...")
    t_train_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t_epoch_start = time.perf_counter()
        for X, y in train_loader:
            X = X.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            optimizer.zero_grad()
            with autocast(**amp_args):
                y_hat = model(X)
                loss  = criterion(y_hat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * X.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)
        epoch_time = time.perf_counter() - t_epoch_start
        print(f"Epoch {epoch:02d}/{epochs} — train_loss: {epoch_loss:.4e} — time: {epoch_time:.2f}s")
    total_train_time = time.perf_counter() - t_train_start
    print(f"Training complete in {total_train_time:.2f}s")

    # 4) Evaluation on test set
    print("Starting evaluation on test set...")
    model.eval()
    preds, trues = [], []
    t_test_start = time.perf_counter()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)
            with autocast(**amp_args):
                out = model(X)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
    test_time = time.perf_counter() - t_test_start
    print(f"Test inference complete in {test_time:.2f}s")

    # 5) Concatenate & inverse-scale
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    y_pred_inv = data_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(y_pred.shape)
    y_true_inv = data_scaler.inverse_transform(y_true.reshape(-1,1)).reshape(y_true.shape)

    # 6) Metrics
    if seq2seq:
        mse, mae = compute_metrics_seq2seq(y_true_inv, y_pred_inv)
    else:
        mse, mae = compute_metrics(y_true_inv, y_pred_inv)

    # 7) Cleanup
    del train_loader, test_loader, train_ds, test_ds, model, optimizer, criterion
    torch.cuda.empty_cache()
    gc.collect()

    return mse, mae, y_pred_inv, y_true_inv, history



