from __future__ import annotations

import re
from pathlib import Path

import joblib
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .xcelerator import CLUSTER_FEATURES

TEXT_CANDIDATES = [
    'product_name',
    'product_text',
    'dominant_category',
    'category',
    'top_categories',
]


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄ0-9]+', str(text).lower()) if len(w) > 2]


def _detect_text_column(df: pd.DataFrame) -> str | None:
    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col
    return None


def train_churn_model(x_path, y_path, output_path):
    print('1. Тренування моделі відтоку (XGBoost)...')
    try:
        X = pd.read_csv(x_path)
        y = pd.read_csv(y_path)
    except FileNotFoundError:
        print(f'❌ Помилка: Не знайдено файли {x_path} або {y_path}.')
        return False

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_series = pd.to_numeric(y.iloc[:, 0], errors='coerce').fillna(0).astype(int)

    model = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y_series)

    artifacts = {
        'model': model,
        'features': list(X.columns),
        'model_type': 'XGBoost',
    }
    joblib.dump(artifacts, output_path)
    print(f'   ✅ XGBoost модель відтоку збережена у: {output_path}')
    return True


def train_word2vec_model(data_path, output_path):
    print('2. Тренування семантичної моделі товарів (Word2Vec)...')
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f'❌ Помилка: Не знайдено файл {data_path}.')
        return False

    text_col = _detect_text_column(df)
    if text_col is None:
        print('   ⚠️ У featured_data.csv немає текстових назв товарів. Word2Vec пропущено.')
        return False

    products = df[text_col].dropna().astype(str).unique()
    sentences = [_tokenize(text) for text in products]
    sentences = [sentence for sentence in sentences if sentence]

    if len(sentences) < 5:
        print('   ⚠️ Замало даних для Word2Vec. Пропускаємо.')
        return False

    model = Word2Vec(sentences, vector_size=20, window=3, min_count=1, workers=1, sg=1)
    model.save(str(output_path))
    print(f'   ✅ Word2Vec модель збережена у: {output_path}')
    return True


def train_kmeans_model(customer_data_path, output_path, n_clusters: int = 5):
    print('3. Тренування K-means кластеризації...')
    try:
        df = pd.read_csv(customer_data_path)
    except FileNotFoundError:
        print(f'❌ Помилка: Не знайдено файл {customer_data_path}.')
        return False

    features = [feature for feature in CLUSTER_FEATURES if feature in df.columns]
    if len(features) < 3:
        print('   ⚠️ Недостатньо ознак для кластеризації.')
        return False

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    model.fit(X_scaled)

    artifacts = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'n_clusters': n_clusters,
    }
    joblib.dump(artifacts, output_path)
    print(f'   ✅ K-means модель збережена у: {output_path}')
    return True


if __name__ == '__main__':
    print('=== ЗАПУСК LOYALTY MODELER ===')

    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    x_train_file = 'X_train.csv'
    y_train_file = 'y_train.csv'
    featured_file = 'featured_data.csv'
    customer_file = 'customer_level_data.csv'

    train_churn_model(x_train_file, y_train_file, models_dir / 'churn_model.joblib')
    train_word2vec_model(featured_file, models_dir / 'word2vec.model')
    train_kmeans_model(customer_file, models_dir / 'kmeans_model.joblib')

    print('\nВсі доступні моделі оновлено. Тепер можна запускати main.py.')
