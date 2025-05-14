# models/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os

# Pastikan folder models/ ada
os.makedirs("models", exist_ok=True)

def preprocess_data(df, scaler_path="models/scaler.pkl", encoder_path="models/label_encoder.pkl"):
    # Inisialisasi label encoder
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(['B', 'M'])  # Explicit class order
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)

    # Encode kolom diagnosis
    if df['diagnosis'].dtype == 'object':
        df['diagnosis'] = label_encoder.transform(df['diagnosis'])

    # Drop kolom tidak penting
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')

    # Hapus outlier menggunakan IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df[~((df < (Q1 - 2 * IQR)) | (df > (Q3 + 2 * IQR))).any(axis=1)]

    # Pisahkan fitur dan label
    X = df_clean.drop('diagnosis', axis=1)
    y = df_clean['diagnosis']

    # Standarisasi fitur
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = StandardScaler()
        scaler.fit(X)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

    return X_scaled, y
