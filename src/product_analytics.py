from __future__ import annotations

from collections import Counter
from typing import Optional
import re

import numpy as np
import pandas as pd

try:
    from gensim.models import Word2Vec
    WORD2VEC_AVAILABLE = True
except Exception:
    Word2Vec = None
    WORD2VEC_AVAILABLE = False

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.config import RANDOM_STATE


STOP_WORDS = {
    'and', 'of', 'the', 'a', 'an', 'to', 'for', 'with', 'in', 'on', 'at', 'by',
    'set', 'pack', 'assorted', 'design', 'style', 'single', 'size', 'small',
    'large', 'medium', 'mini', 'jumbo', 'classic', 'retro', 'vintage',
    'red', 'pink', 'white', 'blue', 'green', 'black', 'silver', 'gold',
    'new', 'hot'
}


def _tokenize(text: str) -> list[str]:
    raw_tokens = re.findall(r'[a-zA-Z0-9]+', str(text).lower())
    result = []

    for token in raw_tokens:
        if token.isdigit():
            continue
        if len(token) < 3:
            continue
        if token in STOP_WORDS:
            continue
        result.append(token)

    return result


def prepare_product_text(products: pd.DataFrame) -> pd.DataFrame:
    products = products.copy()

    products['product_name'] = products.get('product_name', pd.Series('', index=products.index)).fillna('').astype(str)
    products['product_description'] = products.get('product_description', pd.Series('', index=products.index)).fillna('').astype(str)

    products['text_for_embedding'] = (
        products['product_name'] + ' ' + products['product_description']
    ).str.strip()

    if 'category' not in products.columns:
        products['category'] = 'unknown'

    products['category'] = products['category'].fillna('unknown').astype(str)
    products['tokens'] = products['text_for_embedding'].apply(_tokenize)
    return products


def train_word2vec(products: pd.DataFrame, vector_size: int = 50) -> Optional[Word2Vec]:
    if not WORD2VEC_AVAILABLE:
        return None

    if products is None or products.empty:
        return None

    prepared = prepare_product_text(products)
    sentences = [tokens for tokens in prepared['tokens'] if tokens]

    if len(sentences) < 5:
        return None

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=1,
        seed=RANDOM_STATE,
    )
    return model


def average_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    valid_vectors = [model.wv[token] for token in tokens if token in model.wv]

    if not valid_vectors:
        return np.zeros(model.vector_size)

    return np.mean(valid_vectors, axis=0)


def _clean_product_name(name: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9\s]+', ' ', str(name))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _short_label_from_name(name: str, max_words: int = 4) -> str:
    cleaned = _clean_product_name(name)
    words = [w.title() for w in cleaned.split() if w][:max_words]
    return ' '.join(words)


def _dominant_category(cluster_df: pd.DataFrame) -> str:
    if 'category' not in cluster_df.columns:
        return 'unknown'

    counter = Counter(
        cluster_df['category'].fillna('unknown').astype(str).tolist()
    )

    if not counter:
        return 'unknown'

    return counter.most_common(1)[0][0]


def _example_products(cluster_df: pd.DataFrame, limit: int = 3) -> str:
    if 'product_name' not in cluster_df.columns:
        return 'n/a'

    counter = Counter()

    for name in cluster_df['product_name'].fillna('').astype(str):
        cleaned = _clean_product_name(name)
        if cleaned:
            counter[cleaned] += 1

    if not counter:
        return 'n/a'

    return ' | '.join([name for name, _ in counter.most_common(limit)])


def _make_semantic_cluster_name(dominant_category: str, example_products: str) -> str:
    first_example = example_products.split(' | ')[0].strip() if example_products != 'n/a' else ''
    short_example = _short_label_from_name(first_example)

    if dominant_category and dominant_category != 'unknown':
        if short_example:
            return f'{dominant_category} — {short_example}'
        return dominant_category

    if short_example:
        return short_example

    return 'Інший кластер'


def cluster_products(products: pd.DataFrame, model: Optional[Word2Vec], n_clusters: int = 5) -> pd.DataFrame:
    if model is None or products is None or products.empty:
        return pd.DataFrame()

    prepared = prepare_product_text(products)
    tokens_list = prepared['tokens'].tolist()

    if len(tokens_list) < 2:
        return pd.DataFrame()

    vectors = np.vstack([
        average_vector(tokens, model)
        for tokens in tokens_list
    ])

    n_clusters = max(2, min(n_clusters, len(prepared)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    prepared['semantic_cluster'] = kmeans.fit_predict(vectors)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(vectors)
    prepared['pca_x'] = coords[:, 0]
    prepared['pca_y'] = coords[:, 1]

    rows = []

    for cluster_id, cluster_df in prepared.groupby('semantic_cluster'):
        dominant_category = _dominant_category(cluster_df)
        example_products = _example_products(cluster_df)
        cluster_name = _make_semantic_cluster_name(dominant_category, example_products)

        rows.append({
            'semantic_cluster': int(cluster_id),
            'dominant_category': dominant_category,
            'example_products': example_products,
            'semantic_cluster_name': cluster_name,
        })

    cluster_meta = pd.DataFrame(rows)

    prepared = prepared.merge(
        cluster_meta,
        on='semantic_cluster',
        how='left'
    )

    return prepared


def customer_category_profile(products: Optional[pd.DataFrame]) -> pd.DataFrame:
    if products is None or products.empty or 'customer_id' not in products.columns:
        return pd.DataFrame(columns=['customer_id', 'top_categories', 'dominant_category', 'purchased_categories_count'])

    work = products.copy()

    if 'category' not in work.columns:
        work['category'] = 'unknown'

    work['category'] = work['category'].fillna('unknown').astype(str)

    grouped = work.groupby('customer_id')['category'].apply(list).reset_index()
    grouped['top_categories'] = grouped['category'].apply(_top_three_categories)
    grouped['dominant_category'] = grouped['top_categories'].apply(
        lambda x: x.split(',')[0].strip() if str(x).strip() else 'unknown'
    )
    grouped['purchased_categories_count'] = grouped['category'].apply(
        lambda items: len(set([str(x) for x in items if pd.notna(x)]))
    )

    return grouped[['customer_id', 'top_categories', 'dominant_category', 'purchased_categories_count']]


def semantic_cluster_summary(product_clusters: pd.DataFrame) -> pd.DataFrame:
    if product_clusters is None or product_clusters.empty or 'semantic_cluster' not in product_clusters.columns:
        return pd.DataFrame(columns=[
            'semantic_cluster_name',
            'products_count',
            'dominant_category',
            'example_products',
        ])

    rows = []

    for _, cluster_df in product_clusters.groupby('semantic_cluster'):
        cluster_name = cluster_df['semantic_cluster_name'].iloc[0]
        dominant_category = cluster_df['dominant_category'].iloc[0]
        example_products = cluster_df['example_products'].iloc[0]

        rows.append({
            'semantic_cluster_name': cluster_name,
            'products_count': int(len(cluster_df)),
            'dominant_category': dominant_category,
            'example_products': example_products,
        })

    result = pd.DataFrame(rows).sort_values(by='products_count', ascending=False).reset_index(drop=True)
    return result


def _top_three_categories(categories: list[str]) -> str:
    counter = Counter([str(c) for c in categories if pd.notna(c)])

    if not counter:
        return 'unknown'

    return ', '.join([name for name, _ in counter.most_common(3)])