import os
from modules.data_preparation import load_and_aggregate_data, preprocess_data
from modules.predictive_model import train_predictive_models
from modules.causal_model import evaluate_causal_effect


def main():
    print("=== Sistem OULAD Predictive & Prescriptive AI ===")
    data_dir = "data"

    # Validasi Folder Data
    if not os.path.exists(data_dir):
        print(f"Error: Folder '{data_dir}' tidak ditemukan.")
        print("Silakan buat folder 'data' dan letakkan file CSV OULAD di dalamnya.")
        return

    try:
        # FASE 0: Persiapan Data
        df_raw = load_and_aggregate_data(data_path=data_dir)
        df_processed = preprocess_data(df_raw)

        print(f"Data siap: {df_processed.shape[0]} baris, {df_processed.shape[1]} kolom/fitur.")

        # FASE 1: Machine Learning (Prediktif)
        print("\n" + "=" * 40)
        print("MEMULAI FASE 1: PREDICTIVE AI (LEARNING)")
        print("=" * 40)
        top_features = train_predictive_models(df_processed)

        # FASE 2: Causal Reasoning (Preskriptif)
        print("\n" + "=" * 40)
        print("MEMULAI FASE 2: CAUSAL AI (REASONING)")
        print("=" * 40)

        if top_features:
            # Mengambil DataFrame asli yang belum di oversample untuk uji kausal
            evaluate_causal_effect(df_processed, top_features)

    except FileNotFoundError as e:
        print(f"\n[Error] File CSV hilang: {e}")
        print(
            "Pastikan semua file (studentInfo.csv, studentVle.csv, vle.csv, studentAssessment.csv) ada di folder 'data'.")


if __name__ == "__main__":
    main()