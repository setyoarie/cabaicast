from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file  # Mengimpor fungsi dan kelas dari library Flask untuk membuat aplikasi web.
import os  # Mengimpor modul os untuk berinteraksi dengan sistem operasi.
import pandas as pd  # Mengimpor library Pandas untuk manipulasi dan analisis data.
import io  # Mengimpor modul io untuk menangani operasi input/output.
import matplotlib  # Mengimpor library Matplotlib untuk visualisasi data.
import tensorflow as tf  # Mengimpor library TensorFlow untuk pembelajaran mesin dan deep learning.
import datetime  # Mengimpor modul datetime untuk menangani tanggal dan waktu.
import numpy as np  # Mengimpor library NumPy untuk komputasi numerik.
import uuid

matplotlib.use('Agg')  # Mengatur backend Matplotlib untuk bekerja tanpa antarmuka grafis (cocok untuk server).
from matplotlib import pyplot as plt  # Mengimpor modul pyplot dari Matplotlib untuk membuat plot dan grafik.
from matplotlib.backends.backend_pdf import PdfPages  # Mengimpor PdfPages dari Matplotlib untuk menyimpan beberapa halaman plot dalam satu file PDF.
from sklearn.preprocessing import MinMaxScaler  # Mengimpor MinMaxScaler dari scikit-learn untuk normalisasi data.
from sklearn.model_selection import train_test_split  # Mengimpor train_test_split dari scikit-learn untuk membagi dataset menjadi set pelatihan dan pengujian.
from sklearn.decomposition import PCA  # Mengimpor PCA dari scikit-learn untuk analisis komponen utama (Principal Component Analysis).

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Menonaktifkan optimisasi oneDNN di TensorFlow untuk menghindari masalah kompatibilitas atau performa.
from tensorflow.keras.models import load_model, Model, Sequential  # Mengimpor fungsi dan kelas dari Keras (bagian dari TensorFlow) untuk membuat dan memuat model pembelajaran mesin.
from tensorflow.keras.utils import register_keras_serializable, custom_object_scope  # Mengimpor utilitas Keras untuk mendaftarkan kelas kustom dan menciptakan ruang lingkup objek kustom.
from tensorflow.keras.layers import LSTM, Dense, Attention, Concatenate, Input, Permute, Multiply, Lambda, Flatten, Activation, RepeatVector, Dropout, Masking, dot  # Mengimpor berbagai lapisan neural network dari Keras untuk membangun model deep learning.
from tensorflow.keras.optimizers import Adam  # Mengimpor optimizer Adam dari Keras untuk digunakan dalam pelatihan model neural network.
from tensorflow.keras import backend as K  # Mengimpor backend dari Keras untuk operasi backend spesifik seperti manipulasi tensor.
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Mengimpor metrik evaluasi dari scikit-learn untuk mengukur kinerja model.
from werkzeug.utils import secure_filename  # Mengimpor secure_filename dari werkzeug untuk mengamankan nama file yang diunggah.

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey' 

# Placeholder untuk predicted_data
predicted_data = None

class Home():
    @staticmethod
    @app.route('/')  # Menentukan rute untuk URL root (halaman utama) dari aplikasi Flask.
    def index():  # Mendefinisikan fungsi untuk menangani permintaan ke URL root.
        return render_template('index.html')  # Mengembalikan dan merender template HTML bernama 'index.html'.

class ManageData():
    @staticmethod
    @app.route('/lihat_data')  # Menentukan rute untuk URL '/managedata' dari aplikasi Flask.
    def lihat_data():
        data_exists = os.path.exists('uploads/data.csv')
        data = None
        if data_exists:
            # Baca CSV dan konversi tanggal menjadi string
            df = pd.read_csv('uploads/data.csv')
            df['Tanggal'] = df['Tanggal'].astype(str)
            data = df.to_dict(orient='records')
        return render_template('managedata.html', data=data)
    
    @staticmethod
    def is_valid_file(filename):  # Mendefinisikan fungsi untuk memeriksa validitas file berdasarkan ekstensi.
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls']  # Memeriksa apakah ada titik dalam nama file dan apakah ekstensi file adalah 'xlsx' atau 'xls'.
    
    @staticmethod
    @app.route('/upload', methods=['POST'])  # Menentukan rute untuk URL '/upload' dari aplikasi Flask dengan metode HTTP POST.
    def upload_file():  # Mendefinisikan fungsi untuk menangani permintaan upload file.
        if 'file' not in request.files:  # Memeriksa apakah tidak ada file yang diunggah dalam permintaan.
            flash('No file part', 'error')  # Menampilkan pesan kesalahan jika tidak ada bagian file.
            return redirect(request.url)  # Mengarahkan ulang ke URL upload.

        file = request.files['file']  # Mendapatkan file yang diunggah dari permintaan.

        if file.filename == '':  # Memeriksa apakah nama file kosong (tidak ada file yang dipilih).
            flash('No selected file', 'error')  # Menampilkan pesan kesalahan jika tidak ada file yang dipilih.
            return redirect(request.url)  # Mengarahkan ulang ke URL upload.

        if file and ManageData.is_valid_file(file.filename):  # Memeriksa apakah file ada dan valid sesuai dengan ekstensi yang diizinkan.
            filename = secure_filename(file.filename)  # Mengamankan nama file untuk mencegah serangan injeksi.
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Menentukan jalur lengkap untuk menyimpan file.
            file.save(filepath)  # Menyimpan file yang diunggah ke jalur yang ditentukan.

            try:
                data = pd.read_excel(filepath)  # Membaca file Excel yang diunggah menggunakan Pandas.

                if set(data.columns) != {'Tanggal', 'Harga'}:  # Memeriksa apakah kolom dalam data sesuai dengan yang diharapkan.
                    raise Exception('Data must have columns "tanggal" and "harga"')  # Menaikkan pengecualian jika kolom tidak sesuai.

                data['Tanggal'] = pd.to_datetime(data['Tanggal'])
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                data.to_csv('uploads/data.csv', index=False)  # Menyimpan data dalam format CSV di direktori 'uploads'.
                
                plt.figure(figsize=(10, 6))  # Membuat figur untuk plot dengan ukuran 10x6 inci.
                plt.plot(data['Tanggal'], data['Harga'], color='blue')  # Membuat plot dari kolom 'Tanggal' dan 'Harga' dalam data.
                plt.xlabel('Tanggal', fontsize=16)  # Memberikan label pada sumbu x.
                plt.ylabel('Harga/Kg', fontsize=16)  # Memberikan label pada sumbu y.
                plt.title('Plot Harga Cabai Rawit Merah di Jawa Barat', fontsize=18)  # Memberikan judul pada plot.
                plt.legend(fontsize=12)  # Menambahkan legenda pada plot.
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.grid(True)  # Menambahkan grid pada plot.

                plot_path = 'static/images/plot.png'  # Menentukan jalur untuk menyimpan gambar plot.
                plt.savefig(plot_path)  # Menyimpan plot sebagai gambar PNG.
                plt.close()  # Menutup figur plot untuk menghemat memori.
            
                flash('File uploaded successfully', 'success')
                return redirect(url_for('lihat_data', data=data))
            except Exception as e:  # Menangkap pengecualian jika terjadi kesalahan saat memproses file.
                flash(f'Error processing file: {str(e)}', 'danger')  # Menampilkan pesan kesalahan.
                return redirect(url_for('lihat_data'))  # Mengarahkan ulang ke URL 'managedata' tanpa mengirimkan data saat terjadi kesalahan.
        else:
            flash('Invalid file format. Please upload an Excel file with columns "tanggal" and "harga"', 'error')  # Menampilkan pesan kesalahan jika format file tidak valid.
            return redirect(request.url)  # Mengarahkan ulang ke URL upload.

class DataPreparation():
    # Define function to apply PCA
    def apply_PCA(X_input, cum_variance, if_apply):  # Mendefinisikan fungsi untuk menerapkan PCA pada input data.
        if if_apply:  # Mengecek apakah PCA harus diterapkan.
            pca = PCA(n_components=cum_variance)  # Membuat objek PCA dengan jumlah komponen berdasarkan varians kumulatif.
            X_pca = pca.fit_transform(X_input)  # Menerapkan PCA pada input data.
            return X_pca  # Mengembalikan data yang telah diterapkan PCA.
        else:
            return np.array(X_input)  # Mengembalikan data asli jika PCA tidak diterapkan.

    # Define function to prepare data with windowing
    def windowing(X_input, y_input, history_size):  # Mendefinisikan fungsi untuk mempersiapkan data dengan teknik windowing.
        data = []  # Inisialisasi list untuk menyimpan data fitur.
        labels = []  # Inisialisasi list untuk menyimpan label.
        for i in range(history_size, len(y_input)):  # Looping melalui data mulai dari ukuran sejarah hingga panjang label.
            data.append(X_input[i - history_size : i, :])  # Menyimpan potongan data fitur dengan ukuran jendela tertentu.
            labels.append(y_input[i])  # Menyimpan label yang sesuai.
        return np.array(data), np.array(labels).reshape(-1, 1)  # Mengembalikan data dan label dalam bentuk array.

    # Define function to reshape data for LSTM model
    def reshaping(X_train):  # Mendefinisikan fungsi untuk mengubah bentuk data agar sesuai dengan model LSTM.
        return X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])  # Mengubah bentuk data menjadi 2D.

# Define Cyclical Learning Rate class
class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):  # Mendefinisikan kelas untuk Cyclical Learning Rate, turunan dari LearningRateSchedule.
    def __init__(self, initial_learning_rate, maximal_learning_rate, step_size):  # Inisialisasi objek dengan learning rate awal, learning rate maksimal, dan ukuran langkah.
        super(CyclicalLearningRate, self).__init__()  # Memanggil konstruktor kelas induk.
        self.initial_learning_rate = tf.cast(initial_learning_rate, tf.float32)  # Mengubah initial_learning_rate menjadi tipe tf.float32.
        self.maximal_learning_rate = tf.cast(maximal_learning_rate, tf.float32)  # Mengubah maximal_learning_rate menjadi tipe tf.float32.
        self.step_size = tf.cast(step_size, tf.float32)  # Mengubah step_size menjadi tipe tf.float32.

    def __call__(self, step):  # Mendefinisikan metode untuk menghitung learning rate berdasarkan langkah pelatihan.
        step = tf.cast(step, tf.float32)  # Mengubah step menjadi tipe tf.float32.
        cycle = tf.floor(1 + step / (2 * self.step_size))  # Menghitung siklus berdasarkan langkah pelatihan.
        x = tf.abs(step / self.step_size - 2 * cycle + 1)  # Menghitung nilai absolut untuk skala siklus.
        scale_factor = tf.maximum(0.0, (1 - x))  # Menghitung faktor skala dengan nilai maksimum 0.0.
        return self.initial_learning_rate + (self.maximal_learning_rate - self.initial_learning_rate) * scale_factor  # Menghitung dan mengembalikan learning rate baru berdasarkan skala faktor.

    def get_config(self):  # Mendefinisikan metode untuk mendapatkan konfigurasi objek.
        return {
            'initial_learning_rate': float(self.initial_learning_rate.numpy()),  # Mengembalikan initial_learning_rate sebagai float.
            'maximal_learning_rate': float(self.maximal_learning_rate.numpy()),  # Mengembalikan maximal_learning_rate sebagai float.
            'step_size': float(self.step_size.numpy())  # Mengembalikan step_size sebagai float.
        }

# Gunakan custom_object_scope untuk mendaftarkan objek khusus
register_keras_serializable('CyclicalLearningRate', CyclicalLearningRate)  # Mendaftarkan kelas CyclicalLearningRate sebagai objek yang dapat diserialisasi di Keras.

class UseModel():
    @staticmethod
    # Define LSTM model builder
    def build_lstm_model(neurons, input_shape):
        model = Sequential()  # Membuat model Sequential.
        model.add(LSTM(units=neurons, return_sequences=True, activation="relu", input_shape=input_shape))  # Menambahkan layer LSTM dengan jumlah neuron tertentu dan input shape.
        model.add(Flatten())  # Menambahkan layer Flatten untuk mengubah input menjadi bentuk 1D.
        model.add(Dense(units=128, activation='relu'))  # Menambahkan layer Dense dengan 128 unit dan aktivasi ReLU.
        model.add(Dropout(0.1))  # Menambahkan layer Dropout untuk mengurangi overfitting dengan dropout rate 10%.
        model.add(Dense(1))  # Menambahkan layer Dense dengan 1 unit untuk output.
        optimizer = Adam(learning_rate=0.001)  # Menggunakan optimizer Adam dengan learning rate 0.001.
        model.compile(optimizer=optimizer, loss='mean_squared_error')  # Mengompilasi model dengan optimizer dan loss function mean squared error.
        return model  # Mengembalikan model yang sudah dibangun.

    # Define LSTM with Attention model builder
    def build_lstm_attention_model(neurons, input_shape):
        inputs = Input(shape=input_shape)  # Mendefinisikan input dengan bentuk tertentu.
        masked = Masking(mask_value=0.)(inputs)  # Menambahkan layer Masking untuk mengabaikan nilai nol dalam input.
        lstm = LSTM(neurons, return_sequences=True)(masked)  # Menambahkan layer LSTM dengan jumlah neuron tertentu dan return sequences True.
        attention = dot([lstm, lstm], axes=[2, 2])  # Menghitung dot product antara output LSTM untuk attention mechanism.
        attention = Dense(input_shape[0], activation='softmax')(attention)  # Menambahkan layer Dense dengan aktivasi softmax untuk mendapatkan perhatian (attention).
        context = dot([attention, lstm], axes=[2, 1])  # Menghitung context vector dengan dot product antara perhatian dan output LSTM.
        flattened = Flatten()(context)  # Mengubah context vector menjadi bentuk 1D.
        output = Dense(1)(flattened)  # Menambahkan layer Dense dengan 1 unit untuk output.
        model = Model(inputs=inputs, outputs=output)  # Membangun model dengan input dan output yang ditentukan.
        optimizer = Adam(learning_rate=0.001)  # Menggunakan optimizer Adam dengan learning rate 0.001.
        model.compile(optimizer=optimizer, loss='mean_squared_error')  # Mengompilasi model dengan optimizer dan loss function mean squared error.
        return model  # Mengembalikan model yang sudah dibangun.

    # Define function to evaluate model
    def evaluate_model(model, X_test, y_test, scaler):  # Mendefinisikan fungsi untuk mengevaluasi model.
        y_pred = model.predict(X_test)  # Menggunakan model untuk memprediksi data uji.
        y_pred_inv = scaler.inverse_transform(y_pred)  # Membalikkan transformasi pada prediksi.
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))  # Membalikkan transformasi pada data uji.
        mae = mean_absolute_error(y_test_inv, y_pred_inv)  # Menghitung Mean Absolute Error (MAE).
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))  # Menghitung Root Mean Squared Error (RMSE).
        mape = np.mean(np.abs((y_test_inv - y_pred_inv) / y_test_inv)) * 100  # Menghitung Mean Absolute Percentage Error (MAPE).
        return mae, rmse, mape  # Mengembalikan nilai MAE, RMSE, dan MAPE.

class TrainingModel():
    @staticmethod
    @app.route('/trainmodel', methods=['GET', 'POST'])  # Mendefinisikan route untuk halaman trainmodel dengan metode GET dan POST.
    def train_model():
        if request.method == 'POST':  # Mengecek apakah metode request adalah POST.
            neurons = int(request.form['neuronOptions'])  # Mengambil nilai jumlah neuron dari form.
            batch_size = int(request.form['batchSizeOptions'])  # Mengambil nilai batch size dari form.
            epochs = int(request.form['epochs'])  # Mengambil nilai epochs dari form.

            data = pd.read_csv('uploads/data.csv')  # Membaca data dari file CSV yang diunggah.
            data['Tanggal'] = pd.to_datetime(data['Tanggal'])  # Mengonversi kolom 'Tanggal' menjadi datetime.
            data.set_index('Tanggal', inplace=True)  # Menjadikan 'Tanggal' sebagai indeks.

            data.interpolate(method='linear', limit_direction='forward', inplace=True)  # Menginterpolasi data yang hilang secara linear.

            y_scaler_actual = MinMaxScaler()  # Menginisialisasi MinMaxScaler untuk skala data.
            train_cutoff = int(0.8 * len(data))  # Menentukan batas untuk data pelatihan (80%).
            y_price_actual = data[['Harga']]  # Mengambil kolom 'Harga' sebagai DataFrame.
            y_scaler_actual.fit(y_price_actual[:train_cutoff])  # Menyesuaikan skala pada data pelatihan.
            actual_norm = y_scaler_actual.transform(y_price_actual)  # Mengubah data menjadi skala normal.

            params_pca = {'cum_variance': 0.8, 'if_apply': True}  # Menentukan parameter untuk PCA.
            X_pca = DataPreparation.apply_PCA(actual_norm, **params_pca)  # Menerapkan PCA pada data.
            hist_size = 24  # Menentukan ukuran sejarah untuk windowing.
            data_norm = np.concatenate((X_pca, actual_norm), axis=1)  # Menggabungkan hasil PCA dan data normal.
            X_train, y_train = DataPreparation.windowing(data_norm[:train_cutoff, :], data_norm[:train_cutoff, -1], hist_size)  # Mempersiapkan data pelatihan.
            X_test, y_test = DataPreparation.windowing(data_norm[train_cutoff:, :], data_norm[train_cutoff:, -1], hist_size)  # Mempersiapkan data pengujian.

            scaler_y = MinMaxScaler()  # Menginisialisasi MinMaxScaler untuk skala data target.
            scaler_y.fit(y_train)  # Menyesuaikan skala pada data target pelatihan.

            initial_learning_rate = 1e-04  # Menentukan nilai awal learning rate.
            maximal_learning_rate = 1e-02  # Menentukan nilai maksimal learning rate.
            steps_per_epoch = len(X_train) // batch_size  # Menghitung jumlah langkah per epoch.
            step_size = 6 * steps_per_epoch  # Menentukan ukuran langkah.

            cyclic_lr = CyclicalLearningRate(initial_learning_rate, maximal_learning_rate, step_size)  # Menggunakan Cyclical Learning Rate.
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)  # Menggunakan callback untuk menghentikan pelatihan dini jika tidak ada peningkatan.
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=cyclic_lr, amsgrad=True)  # Menggunakan optimizer Adam dengan Cyclical Learning Rate.

            input_shape = (X_train.shape[1], X_train.shape[2])  # Menentukan bentuk input untuk model LSTM.

            # Melatih model LSTM
            lstm_model = UseModel.build_lstm_model(neurons, input_shape)
            lstm_model.compile(optimizer=optimizer, loss='mean_absolute_error')
            history_lstm = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                                          epochs=epochs, batch_size=batch_size, callbacks=[callback])
            lstm_model.save('lstm_model.h5', save_traces=True)

            # Melatih model LSTM dengan Attention
            lstm_attention_model = UseModel.build_lstm_attention_model(neurons, input_shape)
            lstm_attention_model.compile(optimizer=optimizer, loss='mae')
            history_lstm_attention = lstm_attention_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                                                              epochs=epochs, batch_size=batch_size, callbacks=[callback])
            lstm_attention_model.save('lstm_attention_model.h5', save_traces=True)

            # Evaluasi model
            lstm_mae, lstm_rmse, lstm_mape = UseModel.evaluate_model(lstm_model, X_test, y_test, scaler_y)
            lstm_attn_mae, lstm_attn_rmse, lstm_attn_mape = UseModel.evaluate_model(lstm_attention_model, X_test, y_test, scaler_y)

            # Menyimpan hasil evaluasi ke session
            session['lstm_mae'] = f"{lstm_mae:.3f}"
            session['lstm_rmse'] = f"{lstm_rmse:.3f}"
            session['lstm_mape'] = f"{lstm_mape:.2f}"
            session['lstm_attn_mae'] = f"{lstm_attn_mae:.3f}"
            session['lstm_attn_rmse'] = f"{lstm_attn_rmse:.3f}"
            session['lstm_attn_mape'] = f"{lstm_attn_mape:.2f}"

            # Membuat plot performa model LSTM dan LSTM dengan Attention
            TrainingModel.plot_performance_lstm(history_lstm, 'LSTM Model Performance')
            TrainingModel.plot_performance_lstm_att(history_lstm_attention, 'LSTM with Attention Model Performance')

            # Tampilkan pesan sukses
            flash('Models have been trained and updated successfully!', 'success')
            return redirect(url_for('train_model'))

        return render_template('trainmodel.html')

    @staticmethod
    def plot_performance_lstm(history_lstm, title):
        plt.figure(figsize=(10, 6))
        plt.plot(history_lstm.history['loss'], label='Training Loss', color='red')
        plt.plot(history_lstm.history['val_loss'], label='Validation Loss', color='blue')
        plt.title(title, fontsize=23)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        image_path = os.path.join('static/images', 'lstm_performance_plot.png')
        plt.savefig(image_path)
        plt.close()

    @staticmethod
    def plot_performance_lstm_att(history_lstm_attention, title):
        plt.figure(figsize=(10, 6))
        plt.plot(history_lstm_attention.history['loss'], label='Training Loss', color='red')
        plt.plot(history_lstm_attention.history['val_loss'], label='Validation Loss', color='blue')
        plt.title(title, fontsize=23)
        plt.xlabel('Epochs', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        image_path = os.path.join('static/images', 'lstm_attention_performance_plot.png')
        plt.savefig(image_path)
        plt.close()

class Peramalan():
    @staticmethod
    @app.route('/peramalan', methods=['GET', 'POST'])
    def peramalan():  
        global predicted_data  
        predicted_data = None  

        if request.method == 'POST':  
            num_days = int(request.form['numDays'])  
            selected_model = request.form.get('modelType', 'LSTM')  

            data = pd.read_csv('uploads/data.csv')  
            data['Tanggal'] = pd.to_datetime(data['Tanggal'])  
            data.set_index('Tanggal', inplace=True)  
            data['Harga'] = data['Harga'].interpolate(method='linear')  

            harga = data['Harga'].values.reshape(-1, 1)  
            scaler = MinMaxScaler(feature_range=(0, 1))  
            harga_scaled = scaler.fit_transform(harga)  

            inputs = harga_scaled[-24:, :]  
            inputs = np.hstack((inputs, inputs))
            X_predict = np.array(inputs).reshape((1, 24, 2))  

            model_paths = {
                'LSTM': 'lstm_model.h5',
                'LSTM_attention': 'lstm_attention_model.h5'
            }
            model_path = model_paths.get(selected_model)

            if model_path and os.path.exists(model_path):
                with custom_object_scope({'CyclicalLearningRate': CyclicalLearningRate}):
                    model = load_model(model_path)
                    
                predictions = []
                current_window = X_predict[0].copy()
                for _ in range(num_days):
                    current_window_expanded = np.expand_dims(current_window, axis=0)
                    try:
                        prediction = model.predict(current_window_expanded)
                        predictions.append(prediction[0, 0])
                        current_window = np.roll(current_window, -1, axis=0)
                        current_window[-1, -1] = prediction
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        break

                if predictions:
                    predictions = np.array(predictions).reshape(-1, 1)
                    predictions_inverse = scaler.inverse_transform(predictions)
                else:
                    predictions_inverse = np.array([]).reshape(-1, 1)

                last_date = data.index[-1]
                future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, num_days+1)]
                predicted_data = pd.DataFrame({
                    'Tanggal': future_dates,
                    'Harga Prediksi': predictions_inverse.flatten()
                })

                plt.figure(figsize=(10, 6))
                plt.plot(data.index, data['Harga'], label='Harga Aktual', color='blue')
                plt.plot(future_dates, predicted_data['Harga Prediksi'], label='Harga Prediksi', color='red')
                # plt.axvline(x=last_date, color='green', linestyle='--', label='Mulai Prediksi')
                plt.xlabel('Tanggal', fontsize=16)
                plt.ylabel('Harga/Kg', fontsize=16)
                plt.title('Harga Cabai Rawit Prediction', fontsize=18)
                plt.legend(fontsize=14)
                plt.grid(True)
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plot_path = os.path.join('static', 'images', 'plot2.png')
                plt.savefig(plot_path)
                plt.close()

                print(predicted_data)
            else:
                flash('Model belum dibuat', 'error')
                return redirect(url_for('peramalan'))
        return render_template('peramalan.html', predicted_data=predicted_data)


@app.route('/download/<file_format>')  # Menentukan rute Flask untuk endpoint download dengan parameter file_format
def download(file_format):  # Mendefinisikan fungsi download yang akan dieksekusi saat rute '/download/<file_format>' diakses
    output = io.BytesIO()  # Membuat objek BytesIO untuk menyimpan data sementara dalam memori

    if file_format == 'csv':  # Mengecek apakah format file yang diminta adalah CSV
        predicted_data.to_csv(output, index=False)  # Menyimpan DataFrame predicted_data ke output dalam format CSV tanpa indeks
        output.seek(0)  # Mengatur ulang posisi pembacaan ke awal buffer
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name='predicted_data.csv')  # Mengirim file CSV sebagai lampiran

    elif file_format == 'excel':  # Mengecek apakah format file yang diminta adalah Excel
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:  # Membuat objek ExcelWriter dengan engine 'xlsxwriter'
            predicted_data.to_excel(writer, index=False, sheet_name='Sheet1')  # Menyimpan DataFrame predicted_data ke Excel tanpa indeks di sheet 'Sheet1'
        output.seek(0)  # Mengatur ulang posisi pembacaan ke awal buffer
        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='predicted_data.xlsx')  # Mengirim file Excel sebagai lampiran

    elif file_format == 'pdf':  # Mengecek apakah format file yang diminta adalah PDF
        with PdfPages(output) as pdf:  # Membuat objek PdfPages untuk menulis ke file PDF
            fig, ax = plt.subplots(figsize=(8, 6))  # Membuat figure dan axis untuk plot dengan ukuran 8x6 inci
            ax.axis('tight')  # Menyesuaikan axis agar tabel pas
            ax.axis('off')  # Mematikan axis
            table = ax.table(cellText=predicted_data.values, colLabels=predicted_data.columns, cellLoc='center', loc='center')  # Membuat tabel dari DataFrame predicted_data
            pdf.savefig(fig, bbox_inches='tight')  # Menyimpan figure ke file PDF
        output.seek(0)  # Mengatur ulang posisi pembacaan ke awal buffer
        return send_file(output, mimetype='application/pdf', as_attachment=True, download_name='predicted_data.pdf')  # Mengirim file PDF sebagai lampiran

    else:
        return "Invalid file format", 400  # Mengembalikan respons kesalahan jika format file tidak valid

if __name__ == '__main__':
    app.run(debug=True, port=8080)
