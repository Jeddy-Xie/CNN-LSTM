import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curves(history, output_dir, filename):
    """
    history: dict with keys 'train_loss' and 'val_loss' (or fallback to 'loss'/'val_loss')
    filename: full filename (e.g. 'cnn_lstm_learning_curve.png')
    """
    os.makedirs(output_dir, exist_ok=True)
    train_loss = history.get('train_loss') or history.get('loss')
    val_loss   = history.get('val_loss')

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss', marker='o')
    plt.plot(val_loss,   label='Val Loss',   marker='o')
    plt.grid(True)
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.close()


def plot_single_forecast(y_true, y_pred, output_dir, filename, sample_index=0):
    """
    y_true, y_pred: arrays of shape (n_samples, timesteps, 1) or (n_samples, timesteps)
    """
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # collapse last singleton dim if present
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true.squeeze(-1)
        y_pred = y_pred.squeeze(-1)

    t_true = y_true[sample_index]
    t_pred = y_pred[sample_index]
    steps = np.arange(len(t_true))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, t_true, label='True',    marker='o')
    plt.plot(steps, t_pred, label='Predicted',marker='x', linestyle='--')
    plt.grid(True)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title(f'Sample {sample_index}: Forecast vs True')
    plt.legend()

    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.close()


def plot_multi_step_forecast(y_pred, y_true, output_dir, filename,
                             sample_idx=0, window_step=5, max_plots=9):
    """
    Visualize multiple sub-windows from a single sample.
    y_true, y_pred: (n_samples, timesteps, 1) or (n_samples, timesteps)
    window_step: length of each mini-window
    max_plots: total number of subplots (arranged √max_plots × √max_plots)
    """
    os.makedirs(output_dir, exist_ok=True)
    pred = np.asarray(y_pred)
    true = np.asarray(y_true)

    # squeeze out singleton last dim if needed
    if true.ndim == 3 and true.shape[-1] == 1:
        true = true.squeeze(-1)
        pred = pred.squeeze(-1)

    total_steps = true.shape[1]
    plots_per_row = int(np.ceil(np.sqrt(max_plots)))
    plt.figure(figsize=(plots_per_row * 4, plots_per_row * 3))

    for i in range(max_plots):
        start = i * window_step
        end   = start + window_step
        if end > total_steps:
            break

        ax = plt.subplot(plots_per_row, plots_per_row, i + 1)
        ax.plot(np.arange(window_step), true[sample_idx, start:end],   marker='o', label='Actual')
        ax.plot(np.arange(window_step), pred[sample_idx, start:end],   marker='x', linestyle='--', label='Pred')
        ax.set_title(f'Steps {start}–{end-1}')
        ax.set_xlabel('Ahead')
        ax.set_ylabel('Value')
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.close()
