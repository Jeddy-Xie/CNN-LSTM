import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from numpy.lib.stride_tricks import sliding_window_view

'''
class TimeSeriesDataset(Dataset):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 window_size: int,
                 forecast_horizon: int,
                 feature_cols: list,
                 target_col: str,
                 return_index: bool = False,
                 seq2seq: bool = False,
                 date_format: str = "%Y/%m/%d %H:%M",
                 pin_memory: bool = False):
        """
        Args:
            dataframe: pd.DataFrame with datetime index.
            window_size: number of past timesteps.
            forecast_horizon: number of future steps to predict.
            feature_cols: columns for input features.
            target_col: column for prediction target.
            return_index: if True, returns date strings alongside tensors.
            seq2seq: if True, targets shape (window_size, horizon), else (horizon,).
            date_format: format for output date strings.
            pin_memory: if True, pin CPU memory for faster GPU transfers.
        """
        num_samples = len(dataframe)
        if window_size + forecast_horizon > num_samples:
            raise ValueError("window_size + forecast_horizon must be <= number of rows in dataframe")

        # Extract as float32 for zero-copy
        feats = dataframe[feature_cols].to_numpy(dtype=np.float32)
        targs = dataframe[target_col].to_numpy(dtype=np.float32)

        # Compute number of valid windows
        n_seq = num_samples - window_size - forecast_horizon + 1

        # Vectorized sliding windows for X
        X_windows = sliding_window_view(feats, window_size, axis=0)
        X_seq = X_windows[:n_seq]  # shape (n_seq, window_size, n_features)
        
        # Transpose to (n_seq, n_features, window_size) for CNN
        X_seq = np.transpose(X_seq, (0, 2, 1))

        # Vectorized sliding windows for targets
        y_windows = sliding_window_view(targs, forecast_horizon, axis=0)
        if seq2seq:
            y_seq_all = sliding_window_view(y_windows, window_size, axis=0)
            y_seq = y_seq_all[:n_seq]
        else:
            # vector forecasting: pick the window immediately after each input
            y_seq = y_windows[window_size-1:window_size-1+n_seq]  # Changed indexing to get exactly one value per window

        # Make copies of the arrays to ensure they are writable
        X_seq = X_seq.copy()
        y_seq = y_seq.copy()

        # Zero-copy conversion to torch.Tensor
        self.X_seq = torch.from_numpy(X_seq)
        self.y_seq = torch.from_numpy(y_seq)
        if pin_memory:
            self.X_seq = self.X_seq.pin_memory()
            self.y_seq = self.y_seq.pin_memory()

        # Precompute date strings once
        self.return_index = return_index
        if return_index:
            dates = dataframe.index.to_series().dt.strftime(date_format).to_numpy()
            # X date: last input step, Y date: last forecast step
            self.x_dates = dates[window_size - 1 : window_size - 1 + n_seq]
            self.y_dates = dates[window_size + forecast_horizon - 1 : window_size + forecast_horizon - 1 + n_seq]

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.return_index:
            return (
                self.X_seq[idx],
                self.y_seq[idx],
                self.x_dates[idx],
                self.y_dates[idx],
            )
        return self.X_seq[idx], self.y_seq[idx]
'''


class TimeSeriesDataset(Dataset):
    """
    Lazily sliced time-series windows for efficient memory usage.

    Args:
        dataframe (pd.DataFrame): DataFrame with datetime index.
        window_size (int): Number of past timesteps.
        forecast_horizon (int): Number of future steps to predict.
        feature_cols (list): Columns to use as input features.
        target_col (str): Column name of the prediction target.
        return_index (bool): If True, return datetime strings for each sample.
        seq2seq (bool): If True, targets shape (window_size, forecast_horizon), else (forecast_horizon,).
        date_format (str): Format string for datetime output.
    """
    def __init__(
        self,
        npy_path: str,
        feat_idx: list[int],
        targ_idx: int,
        window_size: int,
        forecast_horizon: int,
        return_index: bool = False,
        seq2seq: bool = False,
        date_format: str = "%Y/%m/%d %H:%M",
        dates_npy_path: str | None = None,
    ):
        """
        Args:
            npy_path:        Path to your saved array of shape (T, F+1)
            feat_idx:        Which columns to use as input features
            targ_idx:        Which column is the target
            window_size:     Number of past timesteps
            forecast_horizon:Number of future steps to predict
            return_index:    If True, also return date strings
            seq2seq:         If True, produce (window, horizon) targets
            date_format:     How to format dates if return_index
            dates_npy_path:  Optional .npy of T datetime64 values for indexing
        """
        # 1) Memory‐map the full array (never brings whole thing into RAM)
        data = np.load(npy_path, mmap_mode="r")  # shape (T, F+1)
        self.feats = data[:, feat_idx]           # view of features
        self.targs = data[:, targ_idx]           # view of target
        
        # 2) Optional dates for return_index
        self.return_index = return_index
        if return_index:
            if dates_npy_path is None:
                raise ValueError("Must pass dates_npy_path if return_index=True")
            # e.g. an array of dtype='datetime64[ns]' saved earlier
            self.dates = np.load(dates_npy_path, mmap_mode="r")
        
        self.window_size = window_size
        self.horizon     = forecast_horizon
        self.seq2seq     = seq2seq

        # 3) Compute how many windows fit
        total = self.feats.shape[0]
        self.n_seq = total - window_size - forecast_horizon + 1
        if self.n_seq < 1:
            raise ValueError("window + horizon larger than data length")

        self.date_format = date_format

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        # ---- slice inputs ----
        X_win = self.feats[idx : idx + self.window_size]      # shape (W, F)
        X = torch.from_numpy(X_win.T.copy())                  # → (F, W)

        # ---- slice targets ----
        if self.seq2seq:
            # build (W, H) by stacking H-length slices at each t
            t_full = self.targs
            y_list = [t_full[t : t + self.horizon] 
                    for t in range(idx, idx + self.window_size)]
            Y = torch.from_numpy(np.stack(y_list, axis=0).astype(np.float32))
        else:
            # build (H,) vector just after the window
            y_vec = self.targs[
                idx + self.window_size : idx + self.window_size + self.horizon
            ]
            Y = torch.from_numpy(y_vec.astype(np.float32))

        return X, Y
    
def data_load(file_path, x_scaler=None, y_scaler=None):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

    data = pd.read_csv(file_path)
    # Remove unwanted columns
    data.drop(columns=['date', 'symbol'], errors='ignore', inplace=True)

    # Convert 'unix' timestamp to datetime and set as index
    data['date'] = pd.to_datetime(data['unix'], unit='s')
    data.set_index('date', inplace=True)

    # Standardize column names and ordering
    data.rename(
        columns={
            'close': 'y',
            'Volume BTC': 'x1',
            'Volume USD': 'x2',
            'open': 'x3',
            'high': 'x4',
            'low': 'x5'
        },
        inplace=True
    )
    data = data[['y', 'x1', 'x2', 'x3', 'x4', 'x5']]
    data.sort_index(inplace=True)

    # Store the original datetime index.
    original_index = data.index

    # Separate features and target
    X = data.drop(columns=['y'])
    y = data['y'].values.reshape(-1, 1)

    # Apply scaling if required.
    if x_scaler is not None:
        if x_scaler.lower() == 'minmax':
            x_scaler = MinMaxScaler(feature_range=(0, 1))
        elif x_scaler.lower() == 'standard':
            x_scaler = StandardScaler()
        elif x_scaler.lower() == 'robust':
            x_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        else:
            raise ValueError("Unsupported scaler type for X. Use 'minmax', 'standard', or 'robust'.")
        X = x_scaler.fit_transform(X)
    if y_scaler is not None:
        if y_scaler.lower() == 'minmax':
            y_scaler = MinMaxScaler(feature_range=(0, 1))
        elif y_scaler.lower() == 'standard':
            y_scaler = StandardScaler()
        elif y_scaler.lower() == 'robust':
            y_scaler = RobustScaler(quantile_range=(25.0, 75.0))
        else:
            raise ValueError("Unsupported scaler type for y. Use 'minmax', 'standard', or 'robust'.")
        y = y_scaler.fit_transform(y)

    # Create DataFrames from the transformed data with the original datetime index.
    X_df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'], index=original_index)
    y_df = pd.DataFrame(y, columns=['y'], index=original_index)
    data = pd.concat([y_df, X_df], axis=1)

    return data, x_scaler, y_scaler