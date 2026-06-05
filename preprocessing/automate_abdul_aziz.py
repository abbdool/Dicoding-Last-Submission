import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler

def run_preprocessing():
    # 1. Definisi path (relatif terhadap posisi script ini dijalankan)
    raw_data_path = '../churn_raw/churn.csv'
    output_folder = 'churn_preprocessing'
    output_file = os.path.join(output_folder, 'churn_preprocessed.csv')

    # Bikin folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # 2. Memuat Dataset
    print("Memuat dataset mentah...")
    df = pd.read_csv(raw_data_path)

    # 3. Data Cleaning
    print("Membersihkan data...")
    df_clean = df.drop(columns=['customer_id'])
    df_clean = df_clean.drop_duplicates()

    # 4. Encoding Kategorikal
    print("Encoding fitur kategorikal...")
    le = LabelEncoder()
    df_clean['gender'] = le.fit_transform(df_clean['gender'])
    df_clean = pd.get_dummies(df_clean, columns=['country'], drop_first=True)

    # 5. Scaling Numerikal
    print("Scaling fitur numerikal...")
    robust_features = ['balance', 'estimated_salary']
    minmax_features = ['credit_score', 'age', 'tenure', 'products_number']

    robust_scaler = RobustScaler()
    df_clean[robust_features] = robust_scaler.fit_transform(df_clean[robust_features])

    minmax_scaler = MinMaxScaler()
    df_clean[minmax_features] = minmax_scaler.fit_transform(df_clean[minmax_features])

    # 6. Simpan Hasil
    df_clean.to_csv(output_file, index=False)
    print(f"Preprocessing selesai! Data disimpan di: {output_file}")

if __name__ == "__main__":
    run_preprocessing()