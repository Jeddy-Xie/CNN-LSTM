import os
import pandas as pd
from torch.utils.data import DataLoader
from utils.TSDataset import TimeSeriesDataset

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
    train_df,
    test_df,
    window_size: int,
    forecast_horizon: int,
    feature_cols: list,
    target_col: str,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    seq2seq: bool = True
):
    """
    Returns:
      train_loader, test_loader, train_dataset, test_dataset
    """
    # 1) wrap in your optimized Dataset
    train_ds = TimeSeriesDataset(
        dataframe=train_df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        return_index=False,
        seq2seq=seq2seq
    )
    test_ds = TimeSeriesDataset(
        dataframe=test_df,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        feature_cols=feature_cols,
        target_col=target_col,
        return_index=False,
        seq2seq=seq2seq
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    return train_loader, test_loader, train_ds, test_ds
