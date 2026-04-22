import re
from collections import Counter

import pandas as pd
from gensim.models import Word2Vec


TEXT_CANDIDATES = [
    'product_name',
    'product_text',
    'dominant_category',
    'category',
    'top_categories',
]

FALLBACK_CATEGORY = 'Загальний асортимент'


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄ0-9]+', str(text).lower()) if len(w) > 2]


def _detect_text_column(df: pd.DataFrame) -> str | None:
    for col in TEXT_CANDIDATES:
        if col in df.columns:
            return col
    return None


def build_user_profiles(df: pd.DataFrame, model_path: str = 'models/word2vec.model') -> pd.DataFrame:
    print('   -> Профілювання інтересів клієнтів...')

    if df.empty:
        return pd.DataFrame(columns=['customer_id', 'favorite_category'])

    if 'customer_id' not in df.columns:
        raise ValueError("У featured_data немає колонки 'customer_id'.")

    text_col = _detect_text_column(df)
    if text_col is None:
        customers = df[['customer_id']].drop_duplicates().copy()
        customers['favorite_category'] = FALLBACK_CATEGORY
        return customers

    try:
        model = Word2Vec.load(model_path)
        vocab = set(model.wv.index_to_key)
    except FileNotFoundError:
        vocab = None

    def get_favorite_category(items) -> str:
        words = []
        for item in items:
            tokens = _tokenize(item)
            if vocab is not None:
                tokens = [token for token in tokens if token in vocab]
            words.extend(tokens)

        count = Counter(words)
        return count.most_common(1)[0][0] if count else FALLBACK_CATEGORY

    profiles = df.groupby('customer_id')[text_col].apply(list).reset_index()
    profiles['favorite_category'] = profiles[text_col].apply(get_favorite_category)
    return profiles[['customer_id', 'favorite_category']]
