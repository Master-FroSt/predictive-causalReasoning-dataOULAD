import dowhy
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler


def evaluate_causal_effect(df, top_features):
    """
    Estimasi dampak sebab-akibat (Average Treatment Effect) dari fitur hasil ML
    setelah mengendalikan confounder menggunakan DoWhy.
    """
    print("\n=== Memulai Analisis Causal Reasoning ===")
    outcome_var = 'target' # Peluang kelulusan (Y)

    # Z-SCORE NORMALIZATION untuk mengatasi Unit Scale Bias
    df_causal = df.copy()
    scaler = StandardScaler()

    # Deteksi fitur continuous dengan >2 nilai unik
    continuous_cols = [col for col in df_causal.columns if df_causal[col].nunique() > 2 and col != outcome_var]

    if continuous_cols:
        df_causal[continuous_cols] = scaler.fit_transform(df_causal[continuous_cols])
        print(f"[*] Berhasil menerapkan Z-Score Normalization pada {len(continuous_cols)} fitur aktivitas.")

    # IDENTIFIKASI CONFOUNDERS (W)
    confounder_prefixes = ['highest_education_', 'imd_band_', 'gender_']
    confounders = [col for col in df_causal.columns if any(col.startswith(prefix) for prefix in confounder_prefixes)]

    causal_results = {}

    # Menguji setiap Kandidat Aktivitas (Treatment X) ke Outcome Kelulusan (Y)
    for feature in top_features:
        print(f"\n-> Menganalisis efek kausal untuk fitur: '{feature}'")

        # # PEMBENTUKAN CAUSAL GRAPH (DAG) dengan domain knowledge
        confounder_to_target = "\n        ".join([f'"{c}" -> "{outcome_var}";' for c in confounders])
        confounder_to_feature = "\n        ".join([f'"{c}" -> "{feature}";' for c in confounders])

    # Membangun string Causal Graph (DAG) menggunakan format GML (NetworkX)
        # Causal Graph dinamis untuk setiap fitur
        causal_graph = f"""
        digraph {{
            {confounder_to_target}
            {confounder_to_feature}
            {feature} -> target;
        }}
        """

        try:
            # Mendefinisikan Model Kausal
            model = dowhy.CausalModel(
                data=df_causal,
                treatment=feature,
                outcome=outcome_var,
                graph=causal_graph
            )

            # Identifikasi efek Kausal di Graph
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

            # Estimasi Efek Kausal dengan Backdoor Linear Regression
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )

            ate_value = estimate.value
            causal_results[feature] = ate_value
            print(f"   [+] Average Treatment Effect (ATE): {ate_value:.4f}")

            # Uji Robustness Check dengan nilai placebo
            refutation = model.refute_estimate(
                identified_estimand,
                estimate,
                method_name="placebo_treatment_refuter"
            )
            print(f"   [+] Placebo Refutation (Mendekati 0 = Valid): {refutation.new_effect:.4f}")
        except Exception as e:
            print(f"   [-] Gagal membangun model kausal untuk fitur ini (Kemungkinan variabel statis): {e}")

# REKOMENDASI PRESKRIPTIF
    print("\n" + "=" * 40)
    print("KESIMPULAN & REKOMENDASI PRESKRIPTIF")
    print("=" * 40)
    # Filter aktivitas yang meningkatkan kelulusan (ATE positif)
    positive_features = {k: v for k, v in causal_results.items() if v > 0}

    if positive_features:
        # Cari aktivitas dengan ATE terbesar
        best_feature = max(positive_features, key=positive_features.get)
        max_ate = positive_features[best_feature]

        print(
            f"Berdasarkan Causal AI, aktivitas '{best_feature}' memiliki dampak POSITIF TERBESAR (Efek: {max_ate:.4f}).")
        # Rekomendasi tindakan intervensi bagi EWS
        print("\n[REKOMENDASI SISTEM]")
        print(
            f"Sistem EWS harus merekomendasikan mahasiswa berisiko gagal untuk memprioritaskan dan meningkatkan partisipasi pada '{best_feature}'.")
    else:
        print("Tidak ditemukan fitur/aktivitas dengan hubungan kausal positif yang signifikan.")