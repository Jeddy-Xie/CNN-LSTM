import plotly.graph_objects as go
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

def plot_learning_curves(history, output_dir, name):
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss', marker='o', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o', color='red')
    plt.grid(True)
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # Replace the existing learning_curves.png file if it exists
    if os.path.exists(os.path.join(output_dir, f'{name}_learning_curves.png')):
        os.remove(os.path.join(output_dir, f'{name}_learning_curves.png'))
    plt.savefig(os.path.join(output_dir, f'{name}_learning_curves.png'))
    plt.close()

def plot_forecast(last_train, true_future, predicted_future, output_dir, name, scaler=None, forecast_start_index=None):
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
    n_train = len(last_train.tolist())*len(last_train.tolist()[0])
    n_future = len(true_future.tolist()[0])

    if scaler is not None:
        last_train = scaler.inverse_transform(last_train.reshape(-1, 1)).flatten()
        true_future = scaler.inverse_transform(true_future.reshape(-1, 1)).flatten()
        predicted_future = scaler.inverse_transform(predicted_future.reshape(-1, 1)).flatten()
    plt.figure(figsize=(12, 6))
    time_train = np.arange(n_train)
    time_future = np.arange(n_train, n_train + n_future)
    plt.plot(time_train, last_train, label="Historical (Last n training values)", marker='o')
    plt.plot(time_future, true_future, label="Target Values", marker='o', color='blue')
    plt.plot(time_future, predicted_future, label="Forecasted", marker='x', color='red')
    plt.legend()
    plt.title("Forecast vs. Target Values")
    plt.xlabel("Time Step")
    plt.ylabel("Target Value")
    plt.savefig(os.path.join(output_dir, f'{name}_forecast.png'))
    plt.close()

def plot_single_forecast(y_true, y_pred, sample_index=0):
    """
    Plots the true and predicted forecast for a single sample.
    
    Args:
      y_true (array-like): True target values, expected shape (num_samples, time_steps, 1).
      y_pred (array-like): Predicted target values, same shape as y_true.
      sample_index (int): Which sample to plot.
    """
    # Convert to NumPy arrays (if needed) and remove the last singleton dimension
    y_true = np.squeeze(np.asarray(y_true))  # shape becomes (1427, 73)
    y_pred = np.squeeze(np.asarray(y_pred))  # shape becomes (1427, 73)
    
    # Select the specified sample
    true_sample = y_true[sample_index]
    pred_sample = y_pred[sample_index]
    
    # Create time steps on the x-axis. Here, we assume the time steps are 0 to 72.
    time_steps = np.arange(len(true_sample))
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, true_sample, label="True", marker="o", color = 'blue')
    plt.plot(time_steps, pred_sample, label="Predicted", marker="x", linestyle="--", color = 'red')
    plt.xlabel("Time Step")
    plt.ylabel("Target Value")
    plt.title(f"Forecast (red) vs. True (blue) for Sample {sample_index}")
    plt.grid(True)
    plt.show()