import plotly.graph_objects as go
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_learning_curves(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss', marker='o', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o', color='red')
    plt.grid(True)
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_forecast(last_train, true_future, predicted_future, scaler=None, forecast_start_index=None):
    """
    Plots the historical values (last_train), the true future values, 
    and the forecasted values.
    
    Args:
        last_train (array-like): The most recent n training y values.
        true_future (array-like): The next n true target values.
        predicted_future (array-like): The forecasted n target values.
        scaler: If provided, inverse transforms the values before plotting.
        forecast_start_index (int or array-like, optional): 
            The index (or x-axis values) where the forecast starts.
            If None, the forecast axis will be set as np.arange(n_train, n_train+n_future).
    """
    n_train = len(last_train)
    n_future = len(true_future)
    
    # If a scaler is provided, apply inverse transformation
    if scaler is not None:
        last_train = scaler.inverse_transform(last_train.reshape(-1, 1)).flatten()
        true_future = scaler.inverse_transform(true_future.reshape(-1, 1)).flatten()
        predicted_future = scaler.inverse_transform(predicted_future.reshape(-1, 1)).flatten()
    
    # Default: use historical length to set forecast axis
    if forecast_start_index is None:
        forecast_start_index = n_train

    # If forecast_start_index is a scalar, create a range;
    # if it's already an array of x-values, use it directly.
    if np.isscalar(forecast_start_index):
        time_future = np.arange(forecast_start_index, forecast_start_index + n_future)
    else:
        time_future = np.asarray(forecast_start_index)
    
    # Create a time axis for the historical data
    time_train = np.arange(n_train)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_train, last_train, label="Historical (Last n training values)", marker='o')
    plt.plot(time_future, true_future, label="True Future", marker='o', color='blue')
    plt.plot(time_future, predicted_future, label="Forecasted", marker='x', color='red')
    plt.title("Forecast vs. True Future")
    plt.xlabel("Time Step")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(True)
    plt.show()
