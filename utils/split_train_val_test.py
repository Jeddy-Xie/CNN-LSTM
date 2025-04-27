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
    npy_path: str,              
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
    """
    Create train and test dataloaders with proper data splitting.
    
    Args:
        df: DataFrame used only for determining split point
        npy_path: Path to .npy file containing the full dataset
        dates_npy_path: Optional path to dates .npy file
        feat_idx: Indices of feature columns
        targ_idx: Index of target column
        window_size: Number of past timesteps
        forecast_horizon: Number of future steps to predict
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        persistent_workers: Whether to keep workers alive between epochs
        pin_memory: Whether to pin memory for faster GPU transfer
        seq2seq: Whether to use sequence-to-sequence format
    """
    # Split data into train and test
    train_df, test_df = train_test_split(df, train_frac=0.7)
    n_train_rows = len(train_df)
    
    # Load total number of rows from npy file
    total_rows = np.load(npy_path, mmap_mode="r").shape[0]
    
    # Create train dataset (first 70% of data)
    train_ds = TimeSeriesDataset(
        npy_path=npy_path,
        feat_idx=feat_idx,
        targ_idx=targ_idx,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        seq2seq=seq2seq,
        dates_npy_path=dates_npy_path,
        start_idx=0,
        end_idx=n_train_rows
    )
    
    # Create test dataset (last 30% of data)
    test_ds = TimeSeriesDataset(
        npy_path=npy_path,
        feat_idx=feat_idx,
        targ_idx=targ_idx,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        seq2seq=seq2seq,
        dates_npy_path=dates_npy_path,
        start_idx=n_train_rows,
        end_idx=total_rows
    )

    # Only use persistent_workers if num_workers > 0
    persistent_workers = persistent_workers and num_workers > 0

    # Create dataloaders
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
