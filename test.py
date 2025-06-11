import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam  # Contoh: Gunakan optimizer Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Fungsi untuk memuat data dan memprosesnya
def preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Preprocessing data
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    harga = data['Harga'].values

    # Handling missing values
    data.interpolate(method='linear', limit_direction='forward', inplace=True)

    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    harga_scaled = scaler.fit_transform(harga.reshape(-1, 1))

    # Persiapan data untuk LSTM
    X, y = [], []
    time_steps = 10
    for i in range(len(harga_scaled) - time_steps):
        X.append(harga_scaled[i:i + time_steps])
        y.append(harga_scaled[i + time_steps])

    X, y = np.array(X), np.array(y)
    return data, X, y, scaler

# Fungsi untuk membuat prediksi di masa depan dengan model LSTM
def predict_future(last_window, scaler, future_days, model=None):
    if model is None:
        raise ValueError("Model harus dimuat sebelumnya untuk membuat prediksi.")

    predictions = []

    current_window = last_window.copy()

    for _ in range(future_days):
        # Reshape current window for LSTM input
        current_window_expanded = current_window.reshape(1, current_window.shape[0], current_window.shape[1])

        # Predict using the model
        prediction_scaled = model.predict(current_window_expanded)

        # Inverse transform to get actual price
        prediction = scaler.inverse_transform(prediction_scaled)

        # Append predicted price to predictions list
        predictions.append(prediction[0, 0])

        # Update current window for next prediction
        current_window = np.roll(current_window, -1, axis=1)
        current_window[0, -1] = prediction_scaled  # Update last value in current window

    return np.array(predictions)

# Memuat dan memproses data
data_filepath = 'uploads/data.csv'
data, X, y, scaler = preprocess_data(data_filepath)

# Harga terakhir yang diketahui
last_known_price = 37620.0
last_known_scaled_price = scaler.transform(np.array(last_known_price).reshape(-1, 1))[0, 0]

# Memperluas jendela terakhir dengan harga yang diketahui terakhir yang diskalakan
last_window_extended = np.append(X[-1][1:], last_known_scaled_price).reshape(1, -1)

# Memuat model LSTM yang telah dilatih
# Untuk contoh, memuat modelnya dimatikan
# model = load_model('lstm_model.h5')

# Compile model dengan optimizer dan loss function yang sesuai
# model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Jumlah hari yang diprediksi ke depan
future_days = 10

try:
    # Prediksi harga di masa depan dengan model LSTM
    # future_lstm_predictions = predict_future(model, last_window_extended, scaler, future_days)
    # Simulasi tanpa memuat model, untuk menunjukkan bagaimana prediksi di masa depan bekerja
    future_lstm_predictions = predict_future(last_window_extended, scaler, future_days)

    # Membuat tanggal di masa depan untuk plotting
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

    # Membuat DataFrame untuk prediksi di masa depan
    future_lstm_df = pd.DataFrame({'Date': future_dates, 'Predicted_Harga': future_lstm_predictions.flatten()})

    # Menyimpan dan mencetak prediksi di masa depan
    print("Future LSTM Predictions:")
    print(future_lstm_df)

    future_lstm_df.to_csv('uploads/future_lstm_predictions.csv', index=False)

    # Plot prediksi di masa depan
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Harga'], label='Actual Harga', color='blue')
    plt.plot(future_lstm_df['Date'], future_lstm_df['Predicted_Harga'], label='Predicted Harga LSTM', color='red')
    plt.title('Actual vs Predicted Harga Cabai')
    plt.xlabel('Date')
    plt.ylabel('Harga')
    plt.axvline(x=last_date, color='gray', linestyle='--', label='Last Known Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('uploads/future_lstm_predictions_plot.png')
    plt.show()

    print("Future LSTM predictions saved and plot generated.")

except ValueError as ve:
    print(f"Error: {ve}")

