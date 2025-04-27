import torch
import numpy as np
from utils.split_train_val_test import create_dataloaders
import pandas as pd
def save_model(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model(model, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def predict_with_model(
    model,
    df: pd.DataFrame,
    new_data_path: str,
    data_scaler,
    feat_idx: list[int],
    window_size: int,
    forecast_horizon: int,
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
):
    """
    Make predictions on new data using a trained model.
    
    Args:
        model: Trained model
        new_data_path: Path to new data .npy file
        data_scaler: Scaler used for the original training data
        feat_idx: Feature indices to use
        window_size: Window size used in training
        forecast_horizon: Number of steps to forecast
    """
    # Device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()

    new_loader = create_dataloaders(
        df=df,
        npy_path=new_data_path,
        dates_npy_path=None,
        feat_idx=feat_idx,
        targ_idx=None,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seq2seq=True
    )

    predictions = []
    with torch.no_grad():
        for X in new_loader:
            X = X.to(device, non_blocking=use_cuda)
            with torch.amp.autocast(device_type='cuda' if use_cuda else 'cpu', enabled=use_cuda):
                pred = model(X)
            predictions.append(pred.cpu().numpy())

    y_pred = np.concatenate(predictions, axis=0)
    
    y_pred_inv = data_scaler.inverse_transform(y_pred.reshape(-1,1)).reshape(y_pred.shape)

    return y_pred_inv