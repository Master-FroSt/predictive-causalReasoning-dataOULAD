import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_aggregate_data(data_path="data"):
    """
    Mentransformasikan struktur relasional dataset menjadi format flat table
    dengan satu baris merepresentasikan satu mahasiswa dalam satu modul.
    """
    print(f"Memuat data...")

    # Load Data
    df_info = pd.read_csv(f"{data_path}/studentInfo.csv")

    # Agregasi VLE (Aktivitas)
    df_student_vle = pd.read_csv(f"{data_path}/studentVle.csv")
    df_vle = pd.read_csv(f"{data_path}/vle.csv")
    vle_merged = pd.merge(df_student_vle, df_vle, on=['id_site', 'code_module', 'code_presentation'])

    # Pivot table menghitung sum_click setiap aktivitas
    vle_pivot = vle_merged.pivot_table(
        index=['id_student', 'code_module', 'code_presentation'],
        columns='activity_type',
        values='sum_click',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Agregasi Penilaian (Assessments)
    df_student_assess = pd.read_csv(f"{data_path}/studentAssessment.csv")
    df_assess = pd.read_csv(f"{data_path}/assessments.csv")
    assess_merged = pd.merge(df_student_assess, df_assess, on='id_assessment')

    # Feature Engineering 'Delay' berdasarkan waktu tugas
    assess_merged['date'] = assess_merged['date'].fillna(assess_merged['date_submitted'])
    assess_merged['submission_delay'] = assess_merged['date_submitted'] - assess_merged['date']

    # Ekstrasi fitur baru sesuai data dan fitur 'Delay'
    assess_agg = assess_merged.groupby(['id_student', 'code_module', 'code_presentation']).agg(
        avg_assessment_score=('score', 'mean'),
        avg_submission_delay=('submission_delay', 'mean'),
        total_assessments_submitted=('id_assessment', 'count')
    ).reset_index()

    # Merge seluruh data
    df_final = pd.merge(df_info, vle_pivot, on=['id_student', 'code_module', 'code_presentation'], how='left')
    df_final = pd.merge(df_final, assess_agg, on=['id_student', 'code_module', 'code_presentation'], how='left')

    # Penanganan Missing Values
    numeric_cols = df_final.select_dtypes(include=['number']).columns
    string_cols = df_final.select_dtypes(exclude=['number']).columns
    df_final[numeric_cols] = df_final[numeric_cols].fillna(0)
    df_final[string_cols] = df_final[string_cols].fillna('Unknown')

    return df_final


def preprocess_data(df, sample_frac=1.0):
    """
    Pemformatan data, Encoding Variabel Kategorikal, dan Binerisasi Target.
    Mencegah bias data dan menyiapkan format yang aman untuk causal graph.
    """
    print(f"Memproses fitur dan label...")

    # Sampling Data untuk mempercepat eksperimen
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    df_clean = df.drop(columns=['id_student', 'code_module', 'code_presentation'])


    # ONE-HOT ENCODING data kategorikal
    cat_columns = ['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability']
    cat_columns = [col for col in cat_columns if col in df_clean.columns]
    df_clean = pd.get_dummies(df_clean, columns=cat_columns, drop_first=True)

    # Konversi nilai boolean hasil dummies menjadi integer (0/1)
    for col in df_clean.columns:
        if df_clean[col].dtype == bool:
            df_clean[col] = df_clean[col].astype(int)


    # Format nama kolom dengan '_'
    df_clean.columns = df_clean.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    # Binerisasi Target sesuai kelulusan mahasiswa
    df_clean['target'] = df_clean['final_result'].apply(lambda x: 1 if x in ['Pass', 'Distinction'] else 0)
    df_clean.drop(columns=['final_result'], inplace=True)

    return df_clean