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

# ── Token normalisation helpers ───────────────────────────────────────────────
# Handles product names where the same word appears glued ("RETROSPOT") and
# split ("RETRO SPOT") — both must map to the same canonical token so Word2Vec
# treats them as the same word.

def _build_token_normalizer(corpus_tokens: list[list[str]]) -> dict[str, str]:
    """
    Build a mapping {variant -> canonical} from a flat vocabulary.

    Strategy (two passes):
      1. Exact prefix-split dedup: if token A equals token B + token C
         concatenated (e.g. 'retrospot' == 'retro' + 'spot'), map the split
         pair to the joined form as canonical.
      2. Fuzzy substring dedup (optional, lightweight): if two tokens share
         >= 80% of their characters via SequenceMatcher, keep the more
         frequent one as canonical.

    We deliberately keep this lightweight — no external libraries required
    (uses only stdlib difflib).
    """
    from difflib import SequenceMatcher
    from collections import Counter as _Counter

    flat: list[str] = [tok for sent in corpus_tokens for tok in sent]
    freq = _Counter(flat)
    vocab = list(freq.keys())

    canon: dict[str, str] = {}

    # Pass 1 — prefix-split dedup
    # For every token of length >= 7, check if it equals the concatenation of
    # any two shorter vocabulary tokens.  If so, the *joined* form wins.
    vocab_set = set(vocab)
    for tok in sorted(vocab, key=len, reverse=True):
        if len(tok) < 6:
            continue
        if tok in canon:
            continue
        for split_at in range(3, len(tok) - 2):
            left, right = tok[:split_at], tok[split_at:]
            if left in vocab_set and right in vocab_set:
                # joined form 'tok' is canonical; map the two halves to it
                if left not in canon:
                    canon[left] = tok
                if right not in canon:
                    canon[right] = tok
                break

    # Pass 2 — fuzzy near-duplicate dedup (only for short tokens <= 12 chars)
    short_vocab = [v for v in vocab if len(v) <= 12 and v not in canon]
    checked: set[frozenset] = set()
    for i, a in enumerate(short_vocab):
        for b in short_vocab[i + 1:]:
            pair = frozenset({a, b})
            if pair in checked:
                continue
            checked.add(pair)
            ratio = SequenceMatcher(None, a, b).ratio()
            if ratio >= 0.82 and a != b:
                # keep the more frequent one as canonical
                winner = a if freq[a] >= freq[b] else b
                loser  = b if winner == a else a
                if loser not in canon:
                    canon[loser] = winner
    return canon


def _apply_normalizer(tokens: list[str], canon: dict[str, str]) -> list[str]:
    return [canon.get(t, t) for t in tokens]

STOP_WORDS = {
    'and', 'of', 'the', 'a', 'an', 'to', 'for', 'with', 'in', 'on', 'at', 'by',
    'set', 'pack', 'assorted', 'design', 'style', 'single', 'size', 'small',
    'large', 'medium', 'mini', 'jumbo', 'classic', 'retro', 'vintage',
    'red', 'pink', 'white', 'blue', 'green', 'black', 'silver', 'gold',
    'new', 'hot'
}

BAD_CATEGORY_VALUES = {
    '',
    'unknown',
    'general assortment',
    'general merchandise',
    'загальний асортимент',
    'невизначена категорія',
    'інший кластер',
    'інше',
    'other',
    'n/a',
    'none',
    'null',
    'nan',
    '<na>',
}


def _tokenize(text: str) -> list[str]:
    """Basic tokenizer. Token normalisation (RETROSPOT -> RETRO SPOT dedup)
    is applied corpus-wide in train_word2vec / cluster_products via the
    _build_token_normalizer helper, not per-sentence here."""
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


def _normalize_category(value) -> str:
    if pd.isna(value):
        return ''

    text = str(value).strip()
    if not text:
        return ''

    if '—' in text:
        text = text.split('—', 1)[0].strip()

    if ' - ' in text:
        text = text.split(' - ', 1)[0].strip()

    if text.lower() in BAD_CATEGORY_VALUES:
        return ''

    return text


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
    raw_sentences = [tokens for tokens in prepared['tokens'] if tokens]
    if len(raw_sentences) < 5:
        return None

    # Build corpus-wide token normaliser to collapse variants like
    # 'retrospot' / 'retro' + 'spot' into a single canonical token,
    # then apply it to every sentence before training.
    canon = _build_token_normalizer(raw_sentences)
    sentences = [_apply_normalizer(sent, canon) for sent in raw_sentences]

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        # Назви товарів короткі (3-7 токенів). window=5 ефективно ловив весь рядок,
        # знищуючи локальний контекст. window=3 даєть кращі сусідства.
        window=3,
        # min_count=1 залишав однократні слова (шум, друкарські помилки).
        # min_count=2 фільтрує hapax legomena, що значно знижує розмір словника та шум.
        min_count=2,
        workers=1,
        seed=RANDOM_STATE,
    )
    # Store the canon map on the model object so cluster_products can reuse it
    model._token_canon = canon
    return model


def average_vector(tokens: list[str], model: Word2Vec) -> np.ndarray:
    # Apply the same token canon used during training (if stored on the model)
    canon = getattr(model, '_token_canon', {})
    normalised = [canon.get(t, t) for t in tokens]
    valid_vectors = [model.wv[token] for token in normalised if token in model.wv]
    if not valid_vectors:
        return np.zeros(model.vector_size)
    return np.mean(valid_vectors, axis=0)


def _clean_product_name(name: str) -> str:
    cleaned = re.sub(r'[^A-Za-z0-9\s]+', ' ', str(name))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _short_label_from_name(name: str, max_words: int = 4) -> str:
    cleaned = _clean_product_name(name)
    words = [word.title() for word in cleaned.split() if word][:max_words]
    return ' '.join(words)


def _dominant_category(cluster_df: pd.DataFrame) -> str:
    if 'category' not in cluster_df.columns:
        return 'Інше'

    cleaned = [_normalize_category(value) for value in cluster_df['category'].fillna('').astype(str)]
    cleaned = [value for value in cleaned if value]

    if not cleaned:
        return 'Інше'

    return Counter(cleaned).most_common(1)[0][0]


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
    dominant_category = _normalize_category(dominant_category)

    if dominant_category:
        if short_example:
            return f'{dominant_category} — {short_example}'
        return dominant_category

    if short_example:
        return short_example

    return 'Інше'


def cluster_products(products: pd.DataFrame, model: Optional[Word2Vec], n_clusters: int = 5) -> pd.DataFrame:
    if model is None or products is None or products.empty:
        return pd.DataFrame()

    prepared = prepare_product_text(products)
    tokens_list = prepared['tokens'].tolist()
    if len(tokens_list) < 2:
        return pd.DataFrame()

    vectors = np.vstack([average_vector(tokens, model) for tokens in tokens_list])

    # Захист від виродженого випадку: якщо більшість векторів — нулі (товари
    # без розпізнаних токенів через min_count=2), KMeans лягає в одну точку.
    # Беремо лише ненульові вектори для оцінки, а решту відносимо в "Інше".
    nonzero_mask = ~(vectors == 0).all(axis=1)
    n_meaningful = int(nonzero_mask.sum())
    if n_meaningful < max(n_clusters, 4):
        return pd.DataFrame()

    n_clusters = max(2, min(n_clusters, n_meaningful))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    # Тренуємо kmeans тільки на ненульових, але призначаємо для всіх
    kmeans.fit(vectors[nonzero_mask])
    prepared['semantic_cluster'] = kmeans.predict(vectors)

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
    prepared = prepared.merge(cluster_meta, on='semantic_cluster', how='left')
    return prepared


def _top_three_categories(categories: list[str]) -> str:
    cleaned = [_normalize_category(category) for category in categories]
    cleaned = [category for category in cleaned if category]

    if not cleaned:
        return 'Інше'

    counts = Counter(cleaned)
    return ', '.join([name for name, _ in counts.most_common(3)])


def customer_category_profile(products: Optional[pd.DataFrame]) -> pd.DataFrame:
    if products is None or products.empty or 'customer_id' not in products.columns:
        return pd.DataFrame(columns=['customer_id', 'top_categories', 'dominant_category', 'purchased_categories_count'])

    work = products.copy()
    if 'category' not in work.columns:
        work['category'] = 'unknown'

    grouped = work.groupby('customer_id')['category'].apply(list).reset_index()
    grouped['top_categories'] = grouped['category'].apply(_top_three_categories)
    grouped['dominant_category'] = grouped['top_categories'].apply(
        lambda value: _normalize_category(value.split(',')[0].strip()) or 'Інше'
    )
    grouped['purchased_categories_count'] = grouped['category'].apply(
        lambda items: len({_normalize_category(item) for item in items if _normalize_category(item)})
    )

    return grouped[['customer_id', 'top_categories', 'dominant_category', 'purchased_categories_count']]


def semantic_cluster_summary(product_clusters: pd.DataFrame) -> pd.DataFrame:
    if product_clusters is None or product_clusters.empty or 'semantic_cluster' not in product_clusters.columns:
        return pd.DataFrame(columns=['semantic_cluster_name', 'products_count', 'dominant_category', 'example_products'])

    rows = []
    for _, cluster_df in product_clusters.groupby('semantic_cluster'):
        rows.append({
            'semantic_cluster_name': cluster_df['semantic_cluster_name'].iloc[0],
            'products_count': int(len(cluster_df)),
            'dominant_category': cluster_df['dominant_category'].iloc[0],
            'example_products': cluster_df['example_products'].iloc[0],
        })

    return pd.DataFrame(rows).sort_values(by='products_count', ascending=False).reset_index(drop=True)
