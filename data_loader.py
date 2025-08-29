import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import os

def load_data(path: str) -> pd.DataFrame:
    """Загружает датасет из Excel."""
    df = pd.read_excel(path)
    print(f"Загружено {df.shape[0]} строк и {df.shape[1]} столбцов")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очищает, нормализует и добавляет новые признаки."""
    number_cols = [
        'age_y', 'child_count', 'workload_norm', 'sport_activity',
        'weight', 'height', 'mean_sleep_dur_last_mnth',
        'mean_hearbeat_dur_last_mnth'
    ]
    cat_cols = ['family_status', 'gender']

    # Заполнение числовых NaN медианами
    for col in number_cols:
        df[col] = df[col].fillna(df[col].median())

    # Заполнение категориальных NaN
    df['family_status'] = df['family_status'].fillna('unknown')
    df = df.dropna()

    # Бинаризация целевой переменной
    df.loc[:, 'label'] = df['label'].apply(lambda x: 0 if x == 'minimum' else 1)

    # Удаление лишних колонок
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)

    return df

def vectorize_text(df: pd.DataFrame, n_components: int = 16) -> pd.DataFrame:
    """TF-IDF + PCA для текстового признака."""
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_text = vectorizer.fit_transform(df['text'])

    pca = PCA(n_components=n_components, random_state=42)
    tfidf_text = pca.fit_transform(tfidf_text.toarray())
    tfidf_text = pd.DataFrame(tfidf_text)

    without_text_columns = ['label', 'family_status', 'gender', 'age_y', 'child_count',
                            'workload_norm', 'sport_activity', 'weight', 'height',
                            'mean_sleep_dur_last_mnth', 'mean_hearbeat_dur_last_mnth']

    final_df = pd.concat([tfidf_text, df[without_text_columns].reset_index()], axis=1)
    return final_df

def split_data(final_df: pd.DataFrame):
    """Делит датасет на train/valid/test."""
    X = final_df.drop('label', axis=1)
    y = final_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def save_splits(X_train, X_valid, X_test, y_train, y_valid, y_test, out_dir="data/processed"):
    """Сохраняет подготовленные датасеты для дальнейшего использования."""
    os.makedirs(out_dir, exist_ok=True)
    X_train.assign(label=y_train).to_csv(f"{out_dir}/train.csv", index=False)
    X_valid.assign(label=y_valid).to_csv(f"{out_dir}/valid.csv", index=False)
    X_test.assign(label=y_test).to_csv(f"{out_dir}/test.csv", index=False)
    print(f"Файлы сохранены в {out_dir}")

if __name__ == "__main__":
    df = load_data("data/raw/final_dataset.xlsx")
    df = preprocess_data(df)
    final_df = vectorize_text(df)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(final_df)
    save_splits(X_train, X_valid, X_test, y_train, y_valid, y_test)