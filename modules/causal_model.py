import dowhy
import pandas as pd
import networkx as nx


def evaluate_causal_effect(df, top_features):
    """
    Tahap Penalaran (Reasoning): Memvalidasi apakah top fitur ML benar-benar memiliki sebab-akibat.
    """
    print("\n=== Memulai Analisis Causal Reasoning ===")

    # Anggap fitur terbaik peringkat ke-1 adalah 'treatment' (misal: 'quiz')
    treatment_var = top_features[4] # ubah untuk menguji fitur lainnya
    outcome_var = 'target'

    # Confounders (Variabel Pengganggu)
    # Logikanya: Latar belakang demografi dapat memengaruhi rajinnya kuis DAN nilai akhir
    confounders = ['highest_education', 'imd_band', 'gender']

    # Membangun string Causal Graph (DAG) menggunakan format GML (NetworkX)
    # Ini adalah representasi eksplisit hipotesis kausal
    causal_graph = """
    digraph {
        highest_education -> target;
        imd_band -> target;
        gender -> target;
        highest_education -> TREATMENT_VAR;
        imd_band -> TREATMENT_VAR;
        gender -> TREATMENT_VAR;
        TREATMENT_VAR -> target;
    }
    """.replace("TREATMENT_VAR", treatment_var)

    # 1. Mendefinisikan Model Causal DoWhy
    print(f"1. Membangun DAG untuk mengeksplorasi hubungan {treatment_var} -> target")
    model = dowhy.CausalModel(
        data=df,
        treatment=treatment_var,
        outcome=outcome_var,
        graph=causal_graph
    )

    # 2. Identifikasi efek
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # 3. Estimasi Efek Kausal (Menggunakan Regresi Linear sebagai Estimator)
    print("2. Mengestimasi Efek Kausal murni...")
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    print(f"-> Nilai Efek Kausal Estimasi: {estimate.value:.4f}")
    if estimate.value > 0:
        print(
            f"   Interpretasi: Peningkatan pada '{treatment_var}' secara kausal terbukti MENINGKATKAN probabilitas lulus.")
    else:
        print(f"   Interpretasi: Peningkatan pada '{treatment_var}' BUKAN penyebab kelulusan (korelasi palsu).")

    # 4. Refutasi (Robustness Check)
    print("\n3. Menguji Keabsahan Kausalitas (Refutasi: Placebo Treatment)...")
    # Mengganti variabel treatment dengan data random untuk melihat apakah model tertipu
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter"
    )
    print(f"Nilai Efek Kausal Placebo (Seharusnya mendekati 0): {refutation.new_effect:.4f}")