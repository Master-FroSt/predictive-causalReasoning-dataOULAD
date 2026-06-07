from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import warnings

# Mengabaikan warning iterasi SVM agar terminal tetap bersih
warnings.filterwarnings("ignore", category=UserWarning)


def train_predictive_models(df):
    """
    Menjalankan proses Machine Learning menggunakan model yang lebih optimal.
    """
    X = df.drop(columns=['target'])
    y = df['target']

    # 1. Random Over-Sampling (ROS) untuk Data Imbalance
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    print(f"Data setelah ROS: {X_res.shape[0]} baris.")

    # MENGGUNAKAN LinearSVC: Ratusan kali lebih cepat dibanding SVC biasa untuk dataset >10.000 baris
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),  # Diberi max_depth mencegah overfitting
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42),
        'Support Vector Machine': LinearSVC(random_state=42, dual=False)
    }

    # Simulasi Feature Selection untuk mengekstrak Top 5 Fitur
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_res, y_res)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = importances.nlargest(5).index.tolist()

    print("\n--- Evaluasi Model (Dengan Top 5 Fitur) ---")
    X_top = X_res[top_features]

    # 2. Iterasi Klasifikasi Sederhana
    for name, model in models.items():
        print(f"Melatih {name}...")
        model.fit(X_top, y_res)
        y_pred = model.predict(X_top)
        acc = accuracy_score(y_res, y_pred)

        print(f"-> {name} Accuracy: {acc:.4f}")
        if name == 'Random Forest':
            print("-> Classification Report Random Forest:")
            print(classification_report(y_res, y_pred))

    print(f"\nFitur Paling Relevan yang ditemukan ML: {top_features}")
    return top_features