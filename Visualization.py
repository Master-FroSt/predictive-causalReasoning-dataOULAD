import matplotlib.pyplot as plt
import networkx as nx


def plot_ml_accuracy():
    """
    Menghasilkan Bar Chart untuk perbandingan Akurasi model Machine Learning.
    Data diambil dari hasil rata-rata 10-Fold CV sebelumnya.
    """
    models = ['Decision Tree', 'Random Forest', 'Linear SVC']
    accuracies = [0.9107, 0.9114, 0.8963]

    # Konfigurasi plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, accuracies, color=['#FF9800', '#4CAF50', '#2196F3'], width=0.6)

    # Kustomisasi tampilan
    ax.set_ylim(0.80, 0.95)  # Dibatasi agar perbedaan terlihat jelas
    ax.set_title('Perbandingan Akurasi Model Prediktif (10-Fold CV)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Skor Akurasi', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Menambahkan label angka di atas setiap batang
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.002, f'{yval:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_ate_values():
    """
    Menghasilkan Bar Chart horizontal untuk membandingkan nilai ATE
    (Average Treatment Effect) dari masing-masing fitur.
    """
    features = ['total_assessments_submitted', 'avg_assessment_score', 'homepage', 'quiz', 'resource']
    ate_values = [0.3494, 0.2916, 0.2109, 0.1571, 0.1210]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features, ate_values, color='#9C27B0')
    ax.invert_yaxis()  # Membalik sumbu Y agar nilai tertinggi ada di atas

    # Kustomisasi tampilan
    ax.set_title('Dampak Kausal (ATE) per 1 Standar Deviasi Peningkatan', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Probabilitas Peningkatan Kelulusan', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Menambahkan label angka di ujung setiap batang
    for bar in bars:
        xval = bar.get_width()
        ax.text(xval + 0.005, bar.get_y() + bar.get_height() / 2, f'{xval:.4f} ({xval * 100:.2f}%)',
                ha='left', va='center', fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_causal_dag(treatment_name='total_assessments_submitted'):
    """
    Menghasilkan Network Graph untuk memvisualisasikan Directed Acyclic Graph (DAG).
    Menggambarkan hubungan antara Confounder, Treatment, dan Outcome.
    """
    G = nx.DiGraph()

    # Mendefinisikan node utama
    confounder = 'Demografi\n(Pendidikan, Gender, IMD)'
    treatment = f'Treatment\n({treatment_name})'
    outcome = 'Outcome\n(Target Kelulusan)'

    # Menambahkan edge (panah penunjuk hubungan)
    G.add_edge(confounder, treatment)
    G.add_edge(confounder, outcome)
    G.add_edge(treatment, outcome)

    # Menentukan posisi koordinat node agar berbentuk segitiga rapi
    pos = {
        confounder: (0, 1),
        treatment: (1, 0),
        outcome: (2, 1)
    }

    plt.figure(figsize=(8, 5))

    # Menggambar DAG menggunakan NetworkX
    nx.draw(G, pos, with_labels=True,
            node_color=['#FFCDD2', '#C8E6C9', '#BBDEFB'],  # Warna node: Merah, Hijau, Biru
            node_size=4000,
            font_size=10,
            font_weight='bold',
            arrowsize=20,
            edge_color='gray',
            width=2)

    plt.title('Visualisasi Causal Graph (DAG)', fontsize=14, fontweight='bold', pad=15)
    plt.margins(0.2)
    plt.show()


# =========================================================
# CARA PENGGUNAAN DI KAGGLE:
# Panggil ketiga fungsi di bawah ini pada cell terakhir Anda
# =========================================================
if __name__ == "__main__":
    print("Mencetak Grafik Akurasi Machine Learning...")
    plot_ml_accuracy()

    print("Mencetak Grafik Directed Acyclic Graph (DAG)...")
    plot_causal_dag()

    print("Mencetak Grafik Dampak Kausal (ATE)...")
    plot_ate_values()