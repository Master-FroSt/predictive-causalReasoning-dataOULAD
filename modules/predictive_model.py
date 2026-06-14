from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
import warnings

# Mengabaikan warning iterasi SVM agar terminal tetap bersih
warnings.filterwarnings("ignore", category=UserWarning)


def train_predictive_models(df):
    """
    Menjalankan Machine Learning dengan 10-Fold Cross Validation.
    Menghindari data leakage dengan melakukan Oversampling HANYA pada data training.
    """
    X = df.drop(columns=['target'])
    y = df['target']

    # Konfigurasi 10-Fold Cross Validation
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    ros = RandomOverSampler(random_state=42)
    # Menyiapkan Scaler (Menyamakan rentang nilai klik dan fitur One-Hot)
    scaler = StandardScaler()

    # MENGGUNAKAN LinearSVC: Ratusan kali lebih cepat dibanding SVC biasa untuk dataset >10.000 baris
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),  # Diberi max_depth mencegah overfitting
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'Support Vector Machine': LinearSVC(random_state=42, dual=False)
    }

    # Untuk menyimpan metrik setiap model
    metrics = {name: {'acc': [], 'prec': [], 'rec': [], 'f1': []} for name in models.keys()}

    # Untuk mengakumulasi feature importances dari Random Forest di setiap fold
    rf_feature_importances = np.zeros(X.shape[1])

    print("\n--- Evaluasi Model dengan 10-Fold Cross Validation ---")

    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        # 1. Pisahkan Train dan Test set
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # 2. Lakukan Random Over-Sampling (ROS) HANYA pada Train Set
        X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

        # 2. FEATURE SCALING (StandardScaler)
        # Scaler dilatih (fit) HANYA pada data latih agar tidak terjadi data leakage
        X_train_res_scaled = scaler.fit_transform(X_train_res)
        X_test_scaled = scaler.transform(X_test)

        # Mengembalikan format ke DataFrame agar kolom tidak hilang
        X_train_res_scaled = pd.DataFrame(X_train_res_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

        # 3. Latih dan Evaluasi setiap model
        for name, model in models.items():
            model.fit(X_train_res_scaled, y_train_res)
            y_pred = model.predict(X_test_scaled)

            metrics[name]['acc'].append(accuracy_score(y_test, y_pred))
            metrics[name]['prec'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics[name]['rec'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics[name]['f1'].append(f1_score(y_test, y_pred, zero_division=0))

            # Kumpulkan nilai importance khusus untuk Random Forest
            if name == 'Random Forest':
                rf_feature_importances += model.feature_importances_

        fold += 1

    # Menampilkan rata-rata metrik evaluasi
    for name in models.keys():
        print(f"\n[{name}] Rata-rata 10-Fold CV:")
        print(f"-> Accuracy : {np.mean(metrics[name]['acc']):.4f}")
        print(f"-> Precision: {np.mean(metrics[name]['prec']):.4f}")
        print(f"-> Recall   : {np.mean(metrics[name]['rec']):.4f}")
        print(f"-> F1-Score : {np.mean(metrics[name]['f1']):.4f}")

    # Mengambil Top 5 fitur berdasarkan rata-rata importances di semua fold
    rf_feature_importances /= 10
    importances_series = pd.Series(rf_feature_importances, index=X.columns)
    top_features = importances_series.nlargest(5).index.tolist()

    print(f"\nFitur Paling Relevan yang ditemukan ML (Kandidat Kausal): {top_features}")
    return top_features