from torch.utils.data import Subset
import torch
import numpy as np
from utils.TSDataset import TimeSeriesDataset
from torch.utils.data import DataLoader

def split_train_val_test(dataset, train_frac=0.7):
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_test  = n_total - n_train
    
    assert n_total == n_test + n_train
    train_indices = range(0, n_train)
    test_indices  = range(n_train, n_total)
    
    train_dataset = dataset.iloc[train_indices].sort_index()
    test_dataset  = dataset.iloc[test_indices].sort_index()

    return train_dataset, test_dataset


def create_datasets(train_df, val_df, test_df, window_size, forecast_horizon, feature_cols, target_col, batch_size=32, shuffle=True):
    """
    Creates TimeSeriesDataset objects, their DataLoaders, and converts the sequences to NumPy arrays.
    
    Args:
        train_df, val_df, test_df: DataFrames containing the preprocessed (and scaled) data.
        window_size: Number of past time steps used as input.
        forecast_horizon: Number of future steps to forecast.
        feature_cols: List of feature column names.
        target_col: Name of the target column.
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle the training DataLoader.
    
    Returns:
        A dictionary with dataset objects, DataLoaders, and NumPy arrays:
          - train_dataset, val_dataset, test_dataset
          - train_loader, val_loader, test_loader
          - X_train, y_train, X_val, y_val, X_test, y_test
    """
    # Create dataset objects using your custom TimeSeriesDataset class.
    train_dataset = TimeSeriesDataset(train_df, window_size, forecast_horizon, feature_cols, target_col)
    val_dataset   = TimeSeriesDataset(val_df, window_size, forecast_horizon, feature_cols, target_col)
    test_dataset  = TimeSeriesDataset(test_df, window_size, forecast_horizon, feature_cols, target_col)
    
    # Convert datasets to NumPy arrays for training with TensorFlow/Keras.
    X_train = train_dataset.X_seq.numpy()   # Shape: (n_samples, window_size, n_features)
    y_train = train_dataset.y_seq.numpy()   # Shape: (n_samples, forecast_horizon)
    X_val   = val_dataset.X_seq.numpy()
    y_val   = val_dataset.y_seq.numpy()
    X_test  = test_dataset.X_seq.numpy()
    y_test  = test_dataset.y_seq.numpy()
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
    }