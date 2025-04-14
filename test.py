import os
import sys

import pandas as pd
import numpy as np

import torch
import tensorflow as tf
import keras

from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from utils.TSDataset import TimeSeriesDataset
from utils.TSDataset import data_load
from utils.plot import *
from utils.split_train_val_test import *
from utils.compute_metric import compute_metrics, append_score
from utils.compute_metric import compute_metrics_seq2seq
from utils.models import cnn_lstm_model_eval

import matplotlib as mpl
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization

project_dir = os.path.dirname(os.path.abspath('__file__'))
project_dir

data_path = os.path.join(project_dir, 'data', 'processed', 'BTC-USD-sample.csv')
# Check if Output Directory Exists
if not os.path.exists(os.path.join(project_dir, 'output')):
    os.makedirs(os.path.join(project_dir, 'output'))
output_dir = os.path.join(project_dir, 'output')

# Load data
data1, x_scaler1, y_scaler1 = data_load(data_path, x_scaler='minmax', y_scaler='minmax')

one_step_forecast_scores = []

forecast_horizons = [1]
train_df, test_df = split_train_val_test(data1, train_frac=0.7)
target_col = 'y'

test_target = test_df[target_col].values

for h in forecast_horizons:
    naive_y_true = []  # will hold the true future values for every forecasting window
    naive_y_pred = []  # will hold the corresponding naive predictions
    
    for i in range(len(test_target) - h):
        # True values: for instance, at time step 0, this gets indices 1 to h (i.e., 1:11 when h=10)
        y_true = test_target[i + 1: i + h + 1]
        # Naive predictions: create an array of length h filled with the value at time step i
        y_pred = np.full((h,), test_target[i])
        
        naive_y_true.append(y_true)
        naive_y_pred.append(y_pred)
    
    naive_y_true = y_scaler1.inverse_transform(naive_y_true)
    naive_y_pred = y_scaler1.inverse_transform(naive_y_pred)
    mse_naive, mae_naive, huber_naive = compute_metrics(naive_y_true, naive_y_pred)
    
    # Print out the performance for this forecast horizon
    record = {
        'model': 'Naive',
        'mse': mse_naive,
        'mae': mae_naive,
        'huber': huber_naive
    }
    append_score(one_step_forecast_scores, record)

import pickle
# Parameter bounds for the hyperparameters you want to optimize.
pbounds = {
    'filters': (16, 64),
    'kernel_size': (2, 6),
    'lstm_units': (16, 64),
    'learning_rate': (1e-5, 1e-1),
    'epochs': (20, 100),
    'window_size': (50, 200)
}

# Instantiate Chosen Hyperparameters
data_frame = data1
scaler = y_scaler1
seq2seq = True
strides = 1
forecast_horizon = 1
feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5']
target_col = 'y'
return_index = True
scaler = y_scaler1

# Define the objective function for Bayesian Optimization
def cnn_lstm_model_eval_hpo(filters, window_size, kernel_size, lstm_units, learning_rate, epochs):
    return cnn_lstm_model_eval(
        data_frame=data_frame,
        feature_cols=feature_cols,
        target_col=target_col,
        return_index=return_index,
        scaler=scaler,
        filters=int(filters),
        window_size= int(window_size),
        kernel_size=int(kernel_size),
        strides=strides,
        lstm_units=int(lstm_units),
        learning_rate=learning_rate,
        epochs=int(epochs),
        seq2seq=seq2seq,
        forecast_horizon=forecast_horizon
    )

print("===== Bayesian Optimization for CNN-LSTM =====")
# Run Bayesian Optimization.
optimizer_bo = BayesianOptimization(
    f=cnn_lstm_model_eval_hpo,
    pbounds=pbounds,
    random_state=42
)
# Load previously stored optimization results if the file exists
# Gives User Input to continue from previous results or start from scratch
user_input = input("Continue from previous results? (y/n): ")

pkl_file = "optimizer_results.pkl"
if os.path.exists(pkl_file) and user_input == 'y':
    with open(pkl_file, "rb") as f:
        prev_results = pickle.load(f)
    for res in prev_results:
        optimizer_bo.register(params=res["params"], target=res["target"])

    print("Loaded previous optimization results.")
else:
    print("No previous optimization results found.")

optimizer_bo.maximize(init_points=1, n_iter=0)

print("===== Optimizer Best Hyperparameters =====")
best_hyperparameters = optimizer_bo.max
print(f"Filters: {best_hyperparameters['params']['filters']}")
print(f"Window Size: {best_hyperparameters['params']['window_size']}")
print(f"Kernel Size: {best_hyperparameters['params']['kernel_size']}")
print(f"LSTM Units: {best_hyperparameters['params']['lstm_units']}")
print(f"Learning Rate: {best_hyperparameters['params']['learning_rate']}")
print(f"Epochs: {best_hyperparameters['params']['epochs']}")

# Save Results Using Pickle
with open("optimizer_results.pkl", "wb") as f:
    pickle.dump(optimizer_bo.res, f)

filters = int(best_hyperparameters['params']['filters'])
window_size = int(best_hyperparameters['params']['window_size'])
kernel_size = int(best_hyperparameters['params']['kernel_size'])
lstm_units = int(best_hyperparameters['params']['lstm_units'])
learning_rate = best_hyperparameters['params']['learning_rate']
epochs = int(best_hyperparameters['params']['epochs'])

cnn_lstm_mse, cnn_lstm_mae, cnn_lstm_huber, cnn_lstm_history = cnn_lstm_model_eval(data_frame = data_frame,
                    feature_cols = feature_cols,
                    target_col = target_col,
                    return_index = return_index,
                    scaler = scaler,
                    filters = filters,
                    window_size = window_size,
                    kernel_size = kernel_size,
                    strides = strides,
                    lstm_units = lstm_units,
                    learning_rate = learning_rate,
                    epochs = epochs,
                    seq2seq = seq2seq,
                    forecast_horizon = forecast_horizon,
                    optimize = False)

record = {
    'model': 'CNN-LSTM',
    'mse': -cnn_lstm_mse,
    'mae': cnn_lstm_mae,
    'huber': cnn_lstm_huber
}
append_score(one_step_forecast_scores, record)

print(f"MSE: {-cnn_lstm_mse}")
print(f"MAE: {cnn_lstm_mae}")
print(f"Huber: {cnn_lstm_huber}")

# Save learning curves to output folder
plot_learning_curves(cnn_lstm_history, output_dir, 'cnn_lstm')

print("========== One Step Forecast Scores ==========")
for score in one_step_forecast_scores:
    print(score)