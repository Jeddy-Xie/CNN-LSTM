from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    huber = tf.keras.losses.Huber()(y_true, y_pred).numpy()
    return mse, mae, huber


def append_score(scores, new_score):
    for i, score in enumerate(scores):
        if score['model'] == new_score['model'] and score['h-step Forecast'] == new_score['h-step Forecast']:
            scores[i] = new_score
            return
    scores.append(new_score)