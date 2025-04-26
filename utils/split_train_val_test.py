import os
import pandas as pd
from torch.utils.data import DataLoader
from utils.TSDataset import TimeSeriesDataset
import numpy as np

def train_test_split(df: pd.DataFrame, train_frac: float = 0.7):
    """
    Splits a DataFrame into train and test folds by row-count.
    """
    n = len(df)
    n_train = int(n * train_frac)
    train_df = df.iloc[:n_train].sort_index()
    test_df  = df.iloc[n_train:].sort_index()
    return train_df, test_df

def create_dataloaders(
    df,                        
    npy_path: str,              # path to full_series.npy
    dates_npy_path: str|None,  
    feat_idx: list,
    targ_idx: int,
    window_size: int,
    forecast_horizon: int,
    batch_size: int,
    num_workers: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    seq2seq: bool = True,
):
    train_df, test_df = train_test_split(df, train_frac=0.7)
    n_train_rows = len(train_df)
    n_test_rows  = len(test_df)

    total_rows = np.load(npy_path, mmap_mode="r").shape[0]
    max_start = total_rows - window_size - forecast_horizon + 1

    train_max_start = max(0, n_train_rows - window_size - forecast_horizon + 1)
    train_max_start = min(train_max_start, max_start)
    
    train_range = (0, train_max_start)
    test_range  = (train_max_start, max_start)

    train_ds = TimeSeriesDataset(
        npy_path=npy_path,
        feat_idx=feat_idx,
        targ_idx=targ_idx,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        seq2seq=seq2seq
    )
    test_ds = TimeSeriesDataset(
        npy_path=npy_path,
        feat_idx=feat_idx,
        targ_idx=targ_idx,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        seq2seq=seq2seq
    )

    # Only use persistent_workers if num_workers > 0
    persistent_workers = persistent_workers and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False
    )

    return train_loader, test_loader, train_ds, test_ds
