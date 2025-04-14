from utils.TSDataset import TimeSeriesDataset
from utils.split_train_val_test import split_train_val_test
from utils.compute_metric import compute_metrics_seq2seq, compute_metrics
import tensorflow as tf
from tensorflow import keras
import numpy as np
def cnn_lstm_model_eval(data_frame, feature_cols, target_col, return_index, scaler, filters, window_size, kernel_size, strides, lstm_units, learning_rate, epochs, seq2seq=True, forecast_horizon=1, optimize = True):
    # Use the forecast_horizon 'h' as desired
    h = forecast_horizon
    
    # Split your dataset (assumed to create appropriate targets for seq-to-seq training)
    train_df, test_df = split_train_val_test(data_frame, train_frac=0.7)
    train_set = TimeSeriesDataset(
        dataframe=train_df, 
        window_size=window_size, 
        forecast_horizon=h,
        feature_cols=feature_cols, 
        target_col=target_col, 
        return_index=return_index,
        seq2seq=seq2seq
    )
    test_set = TimeSeriesDataset(
        dataframe=test_df, 
        window_size=window_size, 
        forecast_horizon=h, 
        feature_cols=feature_cols, 
        target_col=target_col, 
        return_index=return_index,
        seq2seq=seq2seq
    )

    # Retrieve sequences for training and testing
    X_train, y_train = train_set.X_seq, train_set.y_seq
    X_test, y_test = test_set.X_seq, test_set.y_seq
    
    # Clear any previous Keras session to avoid cluttered graphs/models
    keras.backend.clear_session()
    
    if seq2seq:
        model = keras.models.Sequential([
            keras.Input(shape=(window_size, len(feature_cols))),
            keras.layers.Conv1D(
                filters=int(filters), 
                kernel_size=int(kernel_size), 
                strides=strides, 
                padding="same"
            ),
            keras.layers.LSTM(int(lstm_units), return_sequences=True),
            keras.layers.LSTM(int(lstm_units), return_sequences=True),
            keras.layers.TimeDistributed(keras.layers.Dense(h))
        ])
    else:
        model = keras.models.Sequential([
            keras.Input(shape=(window_size, len(feature_cols))),
            keras.layers.Conv1D(
                filters=int(filters), 
                kernel_size=int(kernel_size), 
                strides=strides,
                padding="same"
            ),
            keras.layers.LSTM(int(lstm_units), return_sequences=True),
            keras.layers.LSTM(int(lstm_units), return_sequences=False),
            keras.layers.Dense(h)
        ])
        
    # Compile the model with the chosen learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)
    y_pred = model.predict(X_test)

    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)
    y_true_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)
    
    if seq2seq:
        mse, mae, huber = compute_metrics_seq2seq(y_true_inv, y_pred_inv)
    else:
        mse, mae, huber = compute_metrics(y_true_inv, y_pred_inv)
    if optimize:
        return -mse
    else:
        return np.abs(mse), np.abs(mae), np.abs(huber), history.history