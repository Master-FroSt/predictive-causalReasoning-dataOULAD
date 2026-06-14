import dowhy


def evaluate_causal_effect(df, top_features):
    """
    Tahap Penalaran (Reasoning): Menguji seluruh Top Fitur dari ML
    untuk melihat fitur mana yang memiliki efek kausal paling kuat.
    """
    print("\n=== Memulai Analisis Causal Reasoning ===")
    outcome_var = 'target'

    causal_results = {}
    confounders = ['highest_education', 'imd_band', 'gender']

    # Iterasi untuk menguji semua top fitur
    for feature in top_features:
        print(f"\n-> Menganalisis efek kausal untuk fitur: '{feature}'")

    # Membangun string Causal Graph (DAG) menggunakan format GML (NetworkX)
        # Causal Graph dinamis untuk setiap fitur
        causal_graph = f"""
        digraph {{
            highest_education -> target;
            imd_band -> target;
            gender -> target;
            highest_education -> {feature};
            imd_band -> {feature};
            gender -> {feature};
            {feature} -> target;
        }}
        """

        # 1. Membangun DAG dengan Model Causal DoWhy
        model = dowhy.CausalModel(
            data=df,
            treatment=feature,
            outcome=outcome_var,
            graph=causal_graph
        )

        # 2. Identifikasi efek
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

        # 3. Estimasi Efek Kausal (Menggunakan Regresi Linear sebagai Estimator)
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        ate_value = estimate.value
        causal_results[feature] = ate_value
        print(f"   [+] Average Treatment Effect (ATE): {ate_value:.4f}")

        # 4. Refutasi Cepat (Robustness Check)
        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter"
        )
        print(f"   [+] Placebo Refutation (Mendekati 0 = Valid): {refutation.new_effect:.4f}")

    # REKOMENDASI PRESKRIPTIF
    print("\n" + "=" * 40)
    print("KESIMPULAN & REKOMENDASI PRESKRIPTIF")
    print("=" * 40)
    # Filter aktivitas yang memang terbukti MENINGKATKAN kelulusan (ATE > 0)
    positive_features = {k: v for k, v in causal_results.items() if v > 0}

    if positive_features:
        # Cari fitur dengan dampak kausal tertinggi
        best_feature = max(positive_features, key=positive_features.get)
        max_ate = positive_features[best_feature]

        print(
            f"Berdasarkan Causal AI, aktivitas '{best_feature}' memiliki dampak POSITIF TERBESAR (Efek: {max_ate:.4f}).")
        print("\n[REKOMENDASI SISTEM]")
        print(
            f"Sistem EWS harus merekomendasikan mahasiswa berisiko gagal untuk memprioritaskan dan meningkatkan partisipasi pada '{best_feature}'.")
    else:
        print("Tidak ditemukan fitur/aktivitas dengan hubungan kausal positif yang signifikan.")