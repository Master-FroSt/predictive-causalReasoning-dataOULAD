import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_aggregate_data(data_path="data"):
    """
    Mengambil dan menggabungkan data OULAD.
    """
    print(f"Memuat data... (Filter aktivitas VLE)")

    # 1. Load Data Utama
    df_info = pd.read_csv(f"{data_path}/studentInfo.csv")

    # 2. Agregasi VLE (Aktivitas) dengan Time-Window / Cut-off
    df_student_vle = pd.read_csv(f"{data_path}/studentVle.csv")
    df_vle = pd.read_csv(f"{data_path}/vle.csv")

    # Gabungkan VLE
    vle_merged = pd.merge(df_student_vle, df_vle, on=['id_site', 'code_module', 'code_presentation'])

    # Pivot
    vle_pivot = vle_merged.pivot_table(
        index=['id_student', 'code_module', 'code_presentation'],
        columns='activity_type',
        values='sum_click',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # 3. Penggabungan Seluruh Data (avg_score DIBUANG untuk mencegah Data Leakage)
    df_final = pd.merge(df_info, vle_pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')

    numeric_cols = df_final.select_dtypes(include=['number']).columns
    string_cols = df_final.select_dtypes(exclude=['number']).columns

    df_final[numeric_cols] = df_final[numeric_cols].fillna(0)
    df_final[string_cols] = df_final[string_cols].fillna('Unknown')

    return df_final


def preprocess_data(df, sample_frac=1.0):
    """
    Melakukan encoding dan SAMPLING opsional.
    sample_frac: Persentase data yang digunakan (0.0 sampai 1.0) untuk mempercepat eksperimen.
    """
    print(f"Memproses fitur dan label... (Menggunakan {sample_frac * 100}% data)")

    # Sampling Data untuk mempercepat eksperimen (Stratified by region/result dsb idealnya, tapi random cukup)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    df_clean = df.drop(columns=['id_student', 'code_module', 'code_presentation'])


    # ONE-HOT ENCODING (pd.get_dummies)
    cat_columns = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    cat_columns = [col for col in cat_columns if col in df_clean.columns]

    # drop_first=True untuk mencegah multicollinearity (Dummy Variable Trap)
    df_clean = pd.get_dummies(df_clean, columns=cat_columns, drop_first=True)

    # Label Encoding
    le = LabelEncoder()
    cat_columns = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    # Konversi nilai boolean (True/False) dari hasil dummies menjadi integer (1/0)
    for col in df_clean.columns:
        if df_clean[col].dtype == bool:
            df_clean[col] = df_clean[col].astype(int)

    # Binerisasi Target
    df_clean['target'] = df_clean['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)
    df_clean.drop(columns=['final_result'], inplace=True)

    return df_clean