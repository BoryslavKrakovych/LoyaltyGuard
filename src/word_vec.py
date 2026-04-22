from pathlib import Path
import re
from collections import Counter

import pandas as pd
from gensim.models import Word2Vec


CATEGORY_CANDIDATES = [
    'dominant_category',
    'category',
    'top_categories',
]

TEXT_CANDIDATES = [
    'product_name',
    'product_text',
]

STOPWORDS = {
    'and', 'for', 'the', 'with', 'from',
    'set', 'pack', 'pcs', 'piece', 'pieces',
    'шт', 'набір', 'товар'
}


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r'[a-zA-Zа-яА-ЯіІїЇєЄ0-9]+', str(text).lower()) if len(w) > 2]


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _load_or_train_word2vec(df: pd.DataFrame, text_col: str, model_path: str):
    path = Path(model_path)

    if path.exists():
        model = Word2Vec.load(str(path))
        return set(model.wv.index_to_key)

    texts = df[text_col].dropna().astype(str).tolist()
    sentences = [_tokenize(text) for text in texts]
    sentences = [[token for token in sentence if token not in STOPWORDS] for sentence in sentences]
    sentences = [sentence for sentence in sentences if sentence]

    if len(sentences) < 5:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    model = Word2Vec(
        sentences,
        vector_size=20,
        window=3,
        min_count=1,
        workers=1,
        sg=1,
    )
    model.save(str(path))
    return set(model.wv.index_to_key)


def _extract_real_category(items) -> str | None:
    values = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue

        parts = [p.strip() for p in text.split('|') if p.strip()]
        values.extend(parts if parts else [text])

    if not values:
        return None

    count = Counter(values)
    return count.most_common(1)[0][0]


def build_user_profiles(df: pd.DataFrame, model_path: str = 'models/word2vec.model') -> pd.DataFrame:
    print('   -> Профілювання інтересів клієнтів...')

    if df.empty:
        return pd.DataFrame(columns=['customer_id', 'favorite_category'])

    if 'customer_id' not in df.columns:
        raise ValueError("У featured_data немає колонки 'customer_id'.")

    category_col = _first_existing_column(df, CATEGORY_CANDIDATES)
    if category_col is not None:
        profiles = df.groupby('customer_id')[category_col].apply(list).reset_index()
        profiles['favorite_category'] = profiles[category_col].apply(_extract_real_category)
        profiles = profiles.dropna(subset=['favorite_category']).copy()
        profiles['favorite_category'] = profiles['favorite_category'].astype(str).str.strip()
        profiles = profiles[profiles['favorite_category'] != ''].copy()
        return profiles[['customer_id', 'favorite_category']]

    text_col = _first_existing_column(df, TEXT_CANDIDATES)
    if text_col is None:
        return pd.DataFrame(columns=['customer_id', 'favorite_category'])

    vocab = _load_or_train_word2vec(df, text_col, model_path)

    def get_favorite_category(items) -> str | None:
        words = []
        for item in items:
            tokens = _tokenize(item)
            tokens = [token for token in tokens if token not in STOPWORDS]
            if vocab is not None:
                tokens = [token for token in tokens if token in vocab]
            words.extend(tokens)

        count = Counter(words)
        if not count:
            return None
        return count.most_common(1)[0][0]

    profiles = df.groupby('customer_id')[text_col].apply(list).reset_index()
    profiles['favorite_category'] = profiles[text_col].apply(get_favorite_category)
    profiles = profiles.dropna(subset=['favorite_category']).copy()
    profiles['favorite_category'] = profiles['favorite_category'].astype(str).str.strip()
    profiles = profiles[profiles['favorite_category'] != ''].copy()

    return profiles[['customer_id', 'favorite_category']]