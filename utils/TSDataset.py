import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, window_size: int, forecast_horizon: int,
                 feature_cols: list, target_col: str, return_index: bool = False):
        """
        Args:
            dataframe: DataFrame with scaled data.
            window_size: Number of past time steps for input.
            forecast_horizon: Number of future steps to predict.
            feature_cols: List of feature column names.
            target_col: Name of the target column.
            return_index: If True, return the associated datetime indices.
        """
        self.features = dataframe[feature_cols].values
        self.targets = dataframe[target_col].values
        self.index_array = dataframe.index.values
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self._create_sequences()
    
    def _create_sequences(self):
        X_seq, y_seq, x_dates, y_dates = [], [], [], []
        num_samples = len(self.features)
        for i in range(num_samples - self.window_size - self.forecast_horizon + 1):
            X_seq.append(self.features[i: i + self.window_size])
            y_seq.append(self.targets[i + self.window_size : i + self.window_size + self.forecast_horizon])
            # The last date of the input sequence:
            x_dates.append(self.index_array[i + self.window_size - 1])
            # The last date of the target sequence:
            y_dates.append(self.index_array[i + self.window_size + self.forecast_horizon - 1])
        self.X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
        self.y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)
        self.x_dates = x_dates
        self.y_dates = y_dates
    
    def __len__(self):
        return len(self.X_seq)
    
    def __getitem__(self, idx):
        date_format = "%Y/%m/%d %H:%M"
        x_date_str = pd.to_datetime(self.x_dates[idx]).strftime(date_format)
        y_date_str = pd.to_datetime(self.y_dates[idx]).strftime(date_format)
        return self.X_seq[idx], self.y_seq[idx], x_date_str, y_date_str
    

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