# Import necessary libraries
from sklearn.metrics import mean_absolute_error
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import os

# Define save directory on local drive (e.g., a folder named 'stock_data' in the current directory)
save_dir = "./stock_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Create the directory if it doesn't exist

# Define date range
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2022, 1, 1)
start_date_str = str(start.date())
end_date_str = str(end.date())

# Fetch data for a single stock (e.g., AAPL) for simplicity
ticker = 'ICICIBANK.NS'
file_name = f"{save_dir}/{ticker}_{start_date_str}_to_{end_date_str}.csv"

# Check if file already exists; download only if it doesn't
if not os.path.exists(file_name):
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    if not data.empty:
        print(f"Saving {ticker} data to {file_name}")
        data.to_csv(file_name)
    else:
        print(f"No data found for {ticker}")
else:
    print(f"File for {ticker} already exists, skipping download.")

# Load the dataset
df = pd.read_csv(file_name, index_col='Date', parse_dates=True)

# --- Step 1: Data Preprocessing ---
# Use relevant features: Open, High, Low, Close, Volume
features = ['Open', 'High', 'Low', 'Close', 'Volume']
data = df[features]

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences for time-series forecasting


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 3])  # Predicting 'Close' price (index 3)
    return np.array(X), np.array(y)


seq_length = 60  # Using 60 days of past data to predict the next day
X, y = create_sequences(scaled_data, seq_length)

# Split into training (80%) and validation (20%)
train_size = int(len(X) * 0.5)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# --- Step 2: Model Implementation ---
# RNN Model
rnn_model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(seq_length, len(features))),
    Dense(1)
])

# LSTM Model
lstm_model = Sequential([
    LSTM(50, activation='tanh', input_shape=(seq_length, len(features))),
    Dense(1)
])

# --- Step 3: Customized Loss Function ---


def custom_loss(y_true, y_pred):
    error = y_true - y_pred
    # Penalize large errors more (quadratic term for errors > 1)
    penalty = K.switch(K.abs(error) > 1, K.square(error) * 2, K.square(error))
    return K.mean(penalty)


# Compile models with custom loss
rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss)
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_loss)

# --- Step 4: Training and Evaluation ---
# Train RNN
rnn_history = rnn_model.fit(X_train, y_train, epochs=20, batch_size=32,
                            validation_data=(X_val, y_val), verbose=1)

# Train LSTM
lstm_history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32,
                              validation_data=(X_val, y_val), verbose=1)

# Predictions
rnn_pred = rnn_model.predict(X_val)
lstm_pred = lstm_model.predict(X_val)

# Inverse transform predictions and actual values (only for 'Close')
scaler_close = MinMaxScaler()
scaler_close.fit(df[['Close']])  # Fit scaler only on 'Close'
y_val_transformed = scaler_close.inverse_transform(y_val.reshape(-1, 1))
rnn_pred_transformed = scaler_close.inverse_transform(rnn_pred)
lstm_pred_transformed = scaler_close.inverse_transform(lstm_pred)

# --- Step 5: Visualization ---
# Plot 1: Loss vs. Epochs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['loss'], label='RNN Train Loss')
plt.plot(rnn_history.history['val_loss'], label='RNN Val Loss')
plt.title('RNN Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='LSTM Train Loss')
plt.plot(lstm_history.history['val_loss'], label='LSTM Val Loss')
plt.title('LSTM Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Plot 2: Predicted vs. Actual Prices
plt.figure(figsize=(12, 5))
plt.plot(y_val_transformed, label='Actual Prices', color='blue')
plt.plot(rnn_pred_transformed, label='RNN Predictions',
         color='orange', linestyle='--')
plt.plot(lstm_pred_transformed, label='LSTM Predictions',
         color='green', linestyle='--')
plt.title(f'{ticker} Stock Price: Predicted vs. Actual')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Evaluate performance (e.g., Mean Absolute Error)
rnn_mae = mean_absolute_error(y_val_transformed, rnn_pred_transformed)
lstm_mae = mean_absolute_error(y_val_transformed, lstm_pred_transformed)
print(f"RNN MAE: {rnn_mae:.2f}")
print(f"LSTM MAE: {lstm_mae:.2f}")
