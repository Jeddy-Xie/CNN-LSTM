from utils.compute_metric import compute_metrics, compute_metrics_seq2seq
from utils.models import CNNLSTM, WaveNetLSTM
from utils.split_train_val_test import create_dataloaders
import numpy as np
import torch
import gc
from torch.cuda.amp import autocast, GradScaler
from torch import nn
import time
from tqdm import tqdm
import pandas as pd

def cnn_lstm_model_bo(
    df: pd.DataFrame, npy_path: str, verbose: bool,
    feat_idx: list[int],
    targ_idx: int,
    dates_npy_path: str | None,
    filters: int,
    window_size: int,
    kernel_size: int,
    lstm_units: int,
    learning_rate: float,
    epochs: int,
    strides: int = 1,
    val_every: int = None,
    seq2seq: bool = True,
    forecast_horizon: int = 1,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    PyTorch CNN-LSTM train & eval for Bayesian Optimization using memory-mapped dataset.

    Args:
        df (pd.DataFrame): used only to split rows for train/test.
        npy_path: path to .npy of shape (T, F+1).
        feat_idx: indices of feature columns in the .npy array.
        targ_idx: index of the target column in the .npy array.
        dates_npy_path: optional .npy of datetime64[ns] for return_index.
        filters, kernel_size, strides, lstm_units, learning_rate, epochs: model/HPO params.
        seq2seq: True for seq2seq forecasting.
        forecast_horizon: number of steps ahead.
        optimize: if True, return -MSE for HPO; else return full metrics and model.
        batch_size, num_workers, pin_memory: DataLoader args.
        val_every: if None, will be set to max(1, epochs // 10)
    """
    
    # ── device & AMP flags ─────────────────────────────────────────
    use_cuda   = torch.cuda.is_available()
    device     = torch.device("cuda" if use_cuda else "cpu")
    scaler     = torch.amp.GradScaler(enabled=use_cuda)         
    amp_kwargs = dict(device_type='cuda' if use_cuda else 'cpu', enabled=use_cuda)

    # Set val_every based on epochs if not provided
    if val_every is None:
        val_every = max(1, epochs // 10)  # Validate roughly 10 times during training

    # ── DataLoaders (memory-mapped) ────────────────────────────────
    train_loader, test_loader, train_ds, test_ds = create_dataloaders(
        df,
        npy_path=npy_path,
        dates_npy_path=dates_npy_path,
        feat_idx=feat_idx,
        targ_idx=targ_idx,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=pin_memory and use_cuda, 
        persistent_workers=bool(num_workers and use_cuda), 
        seq2seq=seq2seq
    )

    # ── Model / optimiser ─────────────────────────────────────────
    model = CNNLSTM(
        in_channels=len(feat_idx),
        filters=filters,
        kernel_size=kernel_size,
        stride=strides,
        lstm_units=lstm_units,
        horizon=forecast_horizon,
        seq2seq=seq2seq
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {'train_loss': [], 'val_loss': []}

    # ── Training loop ─────────────────────────────────────────────
    best_val_mse = float('inf')

    try:
        for epoch in range(epochs):
            model.train()
            t_epoch_start = time.perf_counter()    # Initialize epoch start time
            data_time, comp_time, running_loss = 0.0, 0.0, 0.0
            loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                               leave=False, disable=not verbose)
            loader_iter.start_t = time.perf_counter()      # initialise start time

            for X_batch, y_batch in loader_iter:
                t0 = time.perf_counter() # -- fetch time
                data_time += t0 - loader_iter.start_t 

                X_batch = X_batch.to(device, non_blocking=use_cuda)
                y_batch = y_batch.to(device, non_blocking=use_cuda)
 
                optimizer.zero_grad()
                t_fwd = time.perf_counter() # -- forward time

                with torch.amp.autocast(**amp_kwargs):
                    y_pred = model(X_batch)
                    loss = criterion(y_pred, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                comp_time += time.perf_counter() - t_fwd  # -- compute time
                running_loss += loss.item() * X_batch.size(0)
                
                # Store timestap of end-of-loop to  measure next wait-for-data time
                loader_iter.start_t = time.perf_counter()
            history['train_loss'].append(running_loss / len(train_loader.dataset))
            epoch_time = time.perf_counter() - t_epoch_start
            if verbose:
                if (epoch + 1) % 6 == 0:
                    print(
                        f"Epoch {epoch+1:02d} | "
                        f"data: {data_time:6.2f}s | "
                        f"compute: {comp_time:6.2f}s | "
                        f"total: {epoch_time:6.2f}s | "
                        f"train_loss: {history['train_loss'][-1]:6.4f}"
                        f"----- Model Params -----"
                        f"batch_size: {batch_size} | "
                        f"window_size: {window_size} | "
                        f"filters: {filters} | "
                        f"kernel_size: {kernel_size} | "
                        f"strides: {strides} | "
                        f"lstm_units: {lstm_units} | "
                    )

            # Validation
            if (epoch + 1) % val_every == 0:
                t_val_start, val_loss = time.perf_counter(), 0.0
                model.eval()
                with torch.no_grad(), torch.amp.autocast(**amp_kwargs):
                    for X_val, y_val in test_loader:
                        X_val = X_val.to(device, non_blocking=use_cuda)
                        y_val = y_val.to(device, non_blocking=use_cuda)

                        y_pred = model(X_val)
                        val_loss += criterion(y_pred, y_val).item() * X_val.size(0)
                history['val_loss'].append(val_loss / len(test_loader.dataset))
                best_val_mse = min(best_val_mse, history['val_loss'][-1])
                if verbose:
                    print(
                        f"Epoch {epoch+1:02d} | "
                        f"val_loss: {history['val_loss'][-1]:6.4f} | "
                        f"val_time: {time.perf_counter() - t_val_start:6.2f}s"
                    )
    finally:
        # 4) Cleanup
        del train_loader, test_loader, train_ds, test_ds, model, optimizer, criterion
        if use_cuda:
            torch.cuda.empty_cache()
        gc.collect()

    return -best_val_mse