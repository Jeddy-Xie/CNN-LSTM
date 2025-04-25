from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def append_score(scores, new_score):
    for i, score in enumerate(scores):
        if score['model'] == new_score['model'] and score['h-step Forecast'] == new_score['h-step Forecast']:
            scores[i] = new_score
            return
    scores.append(new_score)

def append_model(models, new_model):
    for i, model in enumerate(models):
        if model['model'] == new_model['model'] and model['h-step Forecast'] == new_model['h-step Forecast']:
            models[i] = new_model
            return
    models.append(new_model)


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

def compute_metrics_seq2seq(y_true, y_pred):
    """
    Compute MSE and MAE for sequence-to-sequence predictions.
    
    Args:
        y_true: Ground truth values of shape (batch_size, seq_len, horizon)
        y_pred: Predicted values of shape (batch_size, seq_len, horizon)
    
    Returns:
        mse: Mean squared error
        mae: Mean absolute error
    """
    # Ensure both arrays have the same shape
    assert y_true.shape == y_pred.shape, f"Shapes don't match: y_true {y_true.shape} vs y_pred {y_pred.shape}"
    
    # Flatten both arrays to 1D
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    
    # Compute metrics
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    return mse, mae